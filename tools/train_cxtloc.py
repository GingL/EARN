from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import numpy as np
import json
import h5py
import time
import random
from pprint import pprint

# model
import _init_paths
import evals.utils as model_utils
from opt import parse_opt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from loaders.loader import Loader
from mrcn import inference_no_imdb
from sim_utils import com_simV2
from sim_utils import obj_ann_filter
from sim_utils import sub_ann_filter
import functools
import pdb

from layers.lan_enc import RNNEncoder, PhraseAttention
from layers.lan_dec import RNNDncoder, SubjectDecoder, LocationDecoder, RelationDecoder
from layers.vis_enc import LocationEncoder, SubjectEncoder
from layers.reconstruct_loss import AttributeReconstructLoss, LangLangReconstructLoss, VisualLangReconstructLoss, ReconstructionLoss


# IoU function
def computeIoU(box1, box2):
  # each box is of [x1, y1, w, h]
  inter_x1 = max(box1[0], box2[0])
  inter_y1 = max(box1[1], box2[1])
  inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
  inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

  if inter_x1 < inter_x2 and inter_y1 < inter_y2:
    inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
  else:
    inter = 0
  union = box1[2]*box1[3] + box2[2]*box2[3] - inter
  return float(inter)/union


def eval_split(loader, model, split, opt):

    verbose = opt.get('verbose', True)
    num_sents = opt.get('num_sents', -1)
    assert split != 'train', 'Check the evaluation split.'

    model.eval()

    loader.resetIterator(split)
    loss_sum = 0
    loss_evals = 0
    acc = 0
    predictions = []
    finish_flag = False
    model_time = 0

    while True:
        data = loader.getTestBatch(split, opt)
        att_weights = loader.get_attribute_weights()
        sent_ids = data['sent_ids']
        Feats = data['Feats']
        labels = data['labels']
        enc_labels = data['enc_labels']
        dec_labels = data['dec_labels']
        image_id = data['image_id']
        ann_ids = data['ann_ids']
        att_labels, select_ixs = data['att_labels'], data['select_ixs']
        sim = data['sim']

        for i, sent_id in enumerate(sent_ids):
            enc_label = enc_labels[i:i + 1] # (1, sent_len)
            max_len = (enc_label != 0).sum().data[0]
            enc_label = enc_label[:, :max_len]  # (1, max_len)
            dec_label = dec_labels[i:i + 1]
            dec_label = dec_label[:, :max_len]

            label = labels[i:i + 1]
            max_len = (label != 0).sum().data[0]
            label = label[:, :max_len]  # (1, max_len)

            pool5 = Feats['pool5']
            fc7 = Feats['fc7']
            lfeats = Feats['lfeats']
            dif_lfeats = Feats['dif_lfeats']
            
            cxt_fc7 = Feats['cxt_fc7']
            cxt_lfeats = Feats['cxt_lfeats']
            cxt_lfeats_rel = Feats['cxt_lfeats_rel']
            sub_sim = sim['sub_sim'][i:i+1]
            obj_sim = sim['obj_sim'][i:i+1]
            sub_emb = sim['sub_emb'][i:i+1]
            obj_emb = sim['obj_emb'][i:i+1]

            att_label = att_labels[i:i + 1]
            if i in select_ixs:
                select_ix = torch.LongTensor([0]).cuda()
            else:
                select_ix = torch.LongTensor().cuda()

            tic = time.time()
            # if i ==2:
            #     print('2')
            scores, loss, sub_idx, sub_attn, obj_attn, weights, sub_attn_lan, loc_attn_lan, rel_attn_lan, sub_ann_attn, loc_ann_attn, rel_ann_attn = model(pool5, fc7, lfeats, dif_lfeats, cxt_fc7,
                                 cxt_lfeats, cxt_lfeats_rel, label, enc_label, dec_label, sub_sim, obj_sim,
                                 sub_emb, obj_emb, att_label, select_ix, att_weights)


            scores = scores.squeeze(0)
            sub_idx = sub_idx.squeeze(0)
            loss = loss.data[0].item()


            pred_ix = torch.argmax(scores)

            k = 2
            while sub_idx[pred_ix] == 0:
                maxk, idx = torch.topk(scores, k)
                pred_ix = idx[k - 1].data.cpu()
                k += 1
                if k > scores.size(0):
                    break

            pred_ann_id = ann_ids[pred_ix]

            gd_ix = data['gd_ixs'][i]
            loss_sum += loss
            loss_evals += 1

            pred_box = loader.Anns[pred_ann_id]['box']
            gd_box = data['gd_boxes'][i]

            IoU = computeIoU(pred_box, gd_box)
            if opt['use_IoU'] > 0:
                if IoU >= 0.5:
                    acc += 1
            else:
                if pred_ix == gd_ix:
                    acc += 1

            entry = {}
            entry['image_id'] = image_id
            entry['sent_id'] = sent_id
            entry['sent'] = loader.decode_labels(label.data.cpu().numpy())[0]  # gd-truth sent
            entry['gd_ann_id'] = data['ann_ids'][gd_ix]
            entry['pred_ann_id'] = pred_ann_id
            entry['pred_score'] = scores.tolist()[pred_ix]
            entry['IoU'] = IoU
            entry['ann_ids'] = ann_ids
            entry['sub_attn'] = sub_attn.tolist()[0]
            entry['obj_attn'] = obj_attn.tolist()[0]
            entry['weights'] = weights.tolist()[0]
            entry['sub_attn_lan'] = sub_attn_lan.tolist()[0]
            entry['loc_attn_lan'] = loc_attn_lan.tolist()[0]
            entry['rel_attn_lan'] = rel_attn_lan.tolist()[0]
            entry['sub_ann_attn'] = sub_ann_attn.tolist()[0]
            entry['loc_ann_attn'] = loc_ann_attn.tolist()[0]
            entry['rel_ann_attn'] = rel_ann_attn.tolist()[0]



            predictions.append(entry)
            toc = time.time()
            model_time += (toc - tic)

            if num_sents > 0  and loss_evals >= num_sents:
                finish_flag = True
                break
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        # if ix0== 974:
        #    print(1)
        if verbose:
            print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%, (%.4f), model time (per sent) is %.2fs' % \
                  (split, ix0, ix1, acc*100.0/loss_evals, loss, model_time/len(sent_ids)))
        model_time = 0

        if finish_flag or data['bounds']['wrapped']:
            break

    return loss_sum / loss_evals, acc / loss_evals, predictions

class Normalize_Scale(nn.Module):
    def __init__(self, dim, init_norm=20):
        super(Normalize_Scale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        assert isinstance(bottom, Variable), 'bottom must be variable'

        bottom_normalized = nn.functional.normalize(bottom, p=2, dim=1)
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled

class RelationEncoder(nn.Module):
    def __init__(self, opt):
        super(RelationEncoder, self).__init__()
        self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']
        self.fc7_normalizer = Normalize_Scale(opt['fc7_dim'], opt['visual_init_norm'])
        self.lfeat_normalizer    = Normalize_Scale(5, opt['visual_init_norm'])
        self.fc = nn.Linear(opt['fc7_dim']+5, opt['jemb_dim'])

    def forward(self, cxt_feats, cxt_lfeats, cxt_lfeats_rel, obj_attn, wo_obj_idx):
        # cxt_feats.shape = (ann_num,2048), cxt_lfeats=(ann_num,ann_num,5), obj_attn=(sent_num,ann_num),
        # dist=(ann_num,ann_num,1), wo_obj_idx.shape=(sent_num)

        sent_num = obj_attn.size(0)
        ann_num = cxt_feats.size(0)
        batch = sent_num * ann_num

        # cxt_feats
        cxt_feats = cxt_feats.unsqueeze(0).expand(sent_num, ann_num,
                                                  self.fc7_dim)  # cxt_feats.shape = (sent_num，ann_num,2048)
        obj_attn = obj_attn.unsqueeze(1)  # obj_attn=(sent_num, 1, ann_num)
        cxt_feats = torch.bmm(obj_attn, cxt_feats)  # cxt_feats_fuse.shape = (sent_num，1,2048)
        cxt_feats = self.fc7_normalizer(cxt_feats.contiguous().view(sent_num, -1))
        cxt_feats = cxt_feats.unsqueeze(1).expand(sent_num, ann_num, self.fc7_dim)

        cxt_lfeats = cxt_lfeats.unsqueeze(0).expand(sent_num, ann_num, 5)
        # obj_attn = obj_attn.unsqueeze(1)
        cxt_lfeats = torch.bmm(obj_attn, cxt_lfeats) # (batch, 1, 5)
        cxt_lfeats = self.lfeat_normalizer(cxt_lfeats.contiguous().view(sent_num, -1))
        cxt_lfeats = cxt_lfeats.unsqueeze(1).expand(sent_num, ann_num, 5)

        cxt_lfeats_rel = cxt_lfeats_rel.unsqueeze(0).expand(sent_num, ann_num, ann_num, 5)
        cxt_lfeats_rel = cxt_lfeats_rel.contiguous().view(batch, ann_num, 5)
        obj_attn = obj_attn.unsqueeze(1).expand(sent_num, ann_num, 1, ann_num)
        obj_attn = obj_attn.contiguous().view(batch, 1, ann_num)
        cxt_lfeats_rel = torch.bmm(obj_attn, cxt_lfeats_rel) # (batch, 1, 5)
        cxt_lfeats_rel = self.lfeat_normalizer(cxt_lfeats_rel.squeeze(1))
        cxt_lfeats_rel = cxt_lfeats_rel.view(sent_num, ann_num, -1)

        cxt_feats_fuse = torch.cat([cxt_feats, cxt_lfeats, cxt_lfeats_rel], 2)
        
        return cxt_feats_fuse

class Score(nn.Module):
    def __init__(self, vis_dim, lang_dim, jemb_dim):
        super(Score, self).__init__()

        self.feat_fuse = nn.Sequential(nn.Linear(vis_dim+lang_dim, jemb_dim),
                                      nn.ReLU(),
                                      nn.Linear(jemb_dim, 1))
        self.softmax = nn.Softmax(dim=1)
        self.lang_dim = lang_dim
        self.vis_dim = vis_dim

    def forward(self, visual_input, lang_input):

        sent_num, ann_num = visual_input.size(0), visual_input.size(1)

        lang_input = lang_input.unsqueeze(1).expand(sent_num, ann_num, self.lang_dim)
        lang_input = nn.functional.normalize(lang_input, p=2, dim=2)

        ann_attn = self.feat_fuse(torch.cat([visual_input, lang_input], 2))

        ann_attn = self.softmax(ann_attn.view(sent_num, ann_num))
        ann_attn = ann_attn.unsqueeze(2)

        return ann_attn


class SimAttention(nn.Module):
    def __init__(self, vis_dim, jemb_dim):
        super(SimAttention, self).__init__()
        self.embed_dim = 300
        self.feat_fuse = nn.Sequential(nn.Linear(self.embed_dim+vis_dim, jemb_dim),
                                       nn.ReLU(),
                                       nn.Linear(jemb_dim, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, word_emb, vis_feats):
        sent_num, ann_num  = vis_feats.size(0), vis_feats.size(1)
        word_emb = word_emb.unsqueeze(1).expand(sent_num, ann_num, self.embed_dim)
        sim_attn = self.feat_fuse(torch.cat([word_emb, vis_feats], 2))
        sim_attn = sim_attn.squeeze(2)
        return sim_attn


class KARN(nn.Module):
    def __init__(self, opt):
        super(KARN, self).__init__()
        self.num_layers = opt['rnn_num_layers']
        self.hidden_size = opt['rnn_hidden_size']
        self.num_dirs = 2 if opt['bidirectional'] > 0 else 1
        self.jemb_dim = opt['jemb_dim']
        self.word_vec_size = opt['word_vec_size']
        self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']
        self.sub_filter_type = opt['sub_filter_type']
        self.filter_thr = opt['sub_filter_thr']
        self.dist_pel = opt['dist_pel']
        self.net_type = opt['net_type']

        self.lang_res_weight = opt['lang_res_weight']
        self.vis_res_weight = opt['vis_res_weight']
        self.att_res_weight = opt['att_res_weight']
        self.loss_combined = opt['loss_combined']
        self.loss_divided = opt['loss_divided']
        self.use_weight = opt['use_weight']

        # language rnn encoder
        # language rnn encoder
        self.rnn_encoder = RNNEncoder(vocab_size=opt['vocab_size'],
                                      word_embedding_size=opt['word_embedding_size'],
                                      word_vec_size=opt['word_vec_size'],
                                      hidden_size=opt['rnn_hidden_size'],
                                      bidirectional=opt['bidirectional'] > 0,
                                      input_dropout_p=opt['word_drop_out'],
                                      dropout_p=opt['rnn_drop_out'],
                                      n_layers=opt['rnn_num_layers'],
                                      rnn_type=opt['rnn_type'],
                                      variable_lengths=opt['variable_lengths'] > 0)

        self.weight_fc = nn.Linear(self.num_layers * self.num_dirs *self.hidden_size, 3)

        self.sub_attn = PhraseAttention(self.hidden_size * self.num_dirs)
        self.loc_attn = PhraseAttention(self.hidden_size * self.num_dirs)
        self.rel_attn = PhraseAttention(self.hidden_size * self.num_dirs)

        self.rnn_decoder = RNNDncoder(opt)

        self.sub_encoder = SubjectEncoder(opt)
        self.loc_encoder = LocationEncoder(opt)
        self.rel_encoder = RelationEncoder(opt)

        self.sub_sim_attn = SimAttention(self.pool5_dim + self.fc7_dim, self.jemb_dim)
        self.obj_sim_attn = SimAttention(self.fc7_dim, self.jemb_dim)

        self.sub_score = Score(self.pool5_dim+self.fc7_dim, opt['word_vec_size'],
                               opt['jemb_dim'])
        self.loc_score = Score(25+5, opt['word_vec_size'],
                               opt['jemb_dim'])
        self.rel_score = Score(self.fc7_dim+5+5, opt['word_vec_size'],
                               opt['jemb_dim'])

        self.att_res_weight = opt['att_res_weight']
        self.mse_loss = nn.MSELoss()

        self.sub_decoder = SubjectDecoder(opt)
        self.loc_decoder = LocationDecoder(opt)
        self.rel_decoder = RelationDecoder(opt)

        self.att_res_loss = AttributeReconstructLoss(opt)
        self.vis_res_loss = VisualLangReconstructLoss(opt)
        self.lang_res_loss = LangLangReconstructLoss(opt)
        self.rec_loss = ReconstructionLoss(opt)

        self.feat_fuse = nn.Sequential(
            nn.Linear(self.fc7_dim + self.pool5_dim + 25 + 5 + self.fc7_dim + 5 + 5, opt['jemb_dim']))


    def forward(self, pool5, fc7, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, cxt_lfeats_rel, labels, enc_labels, dec_labels,
                sub_sim, obj_sim, sub_emb, obj_emb, att_labels, select_ixs, att_weights):

        sent_num = pool5.size(0)
        ann_num =  pool5.size(1)
        label_mask = (dec_labels != 0).float()

        # 语言特征
        context, hidden, embedded = self.rnn_encoder(labels) # (sent_num, 10, 1024), (sent_num, 1024), (sent_num, 10, 512)

        weights = F.softmax(self.weight_fc(hidden)) # (sent_num, 3)

        sub_attn_lan, sub_phrase_emb = self.sub_attn(context, embedded, labels) # (sent_num, 10), (sent_num, 512)
        loc_attn_lan, loc_phrase_emb = self.loc_attn(context, embedded, labels) # (sent_num, 10), (sent_num, 512)
        rel_attn_lan, rel_phrase_emb = self.rel_attn(context, embedded, labels) # (sent_num, 10), (sent_num, 512)

        # subject feats
        sub_feats = self.sub_encoder(pool5, fc7)  # (sent_num, ann_num, 2048+1024)
        # subject attention
        sub_attn = self.sub_sim_attn(sub_emb, sub_feats)
        sub_loss = self.mse_loss(sub_attn, sub_sim)
        # 高于阈值的内容
        sub_idx = sub_sim.gt(self.filter_thr)
        # 判断是否会过滤掉全部的subject ann
        all_filterd_idx = (sub_idx.sum(1).eq(0))  # (sent_num)
        # 如果全部过滤掉，则不进行过滤
        sub_idx[all_filterd_idx] = 1
        sub_filtered_idx = sub_idx.eq(0)
        sub_feats[sub_filtered_idx] = 0

        # location feats
        loc_feats = self.loc_encoder(lfeats, dif_lfeats)  # (sent_num, ann_num, 5+25)

        # object attention
        cxt_fc7_att = cxt_fc7.unsqueeze(0).expand(sent_num, ann_num, self.fc7_dim)
        cxt_fc7_att = nn.functional.normalize(cxt_fc7_att, p=2, dim=2)
        obj_attn = self.obj_sim_attn(obj_emb, cxt_fc7_att)
        wo_obj_idx = obj_sim.sum(1).eq(0)
        obj_attn[wo_obj_idx] = 0
        obj_loss = self.mse_loss(obj_attn, obj_sim)
        # object feats
        rel_feats = self.rel_encoder(cxt_fc7, cxt_lfeats, cxt_lfeats_rel, obj_attn, wo_obj_idx)  # (sent_num, ann_num, 2048+5) (sent_num, ann_num)
        # dist = 100 / (dist + 100)

        sub_ann_attn = self.sub_score(sub_feats, sub_phrase_emb)  # (sent_num, ann_num, 1)
        loc_ann_attn = self.loc_score(loc_feats, loc_phrase_emb)  # (sent_num, ann_num, 1)
        rel_ann_attn = self.rel_score(rel_feats, rel_phrase_emb)  # (sent_num, ann_num, 1)

        weights_expand = weights.unsqueeze(1).expand(sent_num, ann_num, 3)
        total_ann_score = (weights_expand * torch.cat([sub_ann_attn, loc_ann_attn, rel_ann_attn], 2)).sum(2)  # (sent_num, ann_num)
        # if self.dist_pel > 0:
        #     total_ann_score = total_ann_score * dist

        sub_phrase_recons = self.sub_decoder(sub_feats, total_ann_score)  # (sent_num, 512)
        loc_phrase_recons = self.loc_decoder(loc_feats, total_ann_score)  # (sent_num, 512)
        rel_phrase_recons = self.rel_decoder(rel_feats, total_ann_score)  # (sent_num, 512)

        loss = 0

        # 第一次输出时，发现vis_res_loss=133.2017, att_res_loss = 4.3362, lan_rec_loss = 7.6249
        if self.vis_res_weight > 0:
            vis_res_loss = self.vis_res_loss(sub_phrase_emb, sub_phrase_recons, loc_phrase_emb,
                                             loc_phrase_recons, rel_phrase_emb, rel_phrase_recons, weights)
            loss = self.vis_res_weight * vis_res_loss

        if self.lang_res_weight > 0:
            lang_res_loss = self.lang_res_loss(sub_phrase_emb, loc_phrase_emb, rel_phrase_emb, enc_labels,
                                               dec_labels)

            loss += self.lang_res_weight * lang_res_loss

            # loss += torch.log(torch.div(lang_res_loss, vis_res_loss))
        # combined_loss
        loss = self.loss_divided * loss

        ann_score = total_ann_score.unsqueeze(1)
        fuse_feats = torch.cat([sub_feats, loc_feats, rel_feats], 2)  # (sent_num, ann_num, 2048+1024+512+512)

        fuse_feats = torch.bmm(ann_score, fuse_feats)
        fuse_feats = fuse_feats.squeeze(1)
        fuse_feats = self.feat_fuse(fuse_feats)
        rec_loss = self.rec_loss(fuse_feats, enc_labels, dec_labels)
        loss += self.loss_combined * rec_loss

        loss = loss + sub_loss + obj_loss

        if self.att_res_weight > 0:
            # attribute_feats.shape = (12,12,512),total_ann_score.shape = (12,12), att_labels.shape = (12,50), att_weights.shape = (50)
            att_scores, att_res_loss = self.att_res_loss(sub_feats, total_ann_score, att_labels, select_ixs, att_weights)
            loss += self.att_res_weight * att_res_loss

        return total_ann_score, loss, sub_idx, sub_attn, obj_attn, weights, sub_attn_lan, loc_attn_lan, rel_attn_lan, sub_ann_attn, loc_ann_attn, rel_ann_attn


# box functions
def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

class DataLoader(Loader):

    def __init__(self, data_json, data_h5, sub_obj_wds, similarity, opt):
        # parent loader instance
        Loader.__init__(self, data_json, sub_obj_wds, similarity, data_h5)
        # prepare attributes
        self.att_to_ix = self.info['att_to_ix']
        self.ix_to_att = {ix: wd for wd, ix in self.att_to_ix.items()}
        self.num_atts = len(self.att_to_ix)
        self.att_to_cnt = self.info['att_to_cnt']
        self.vocab_file = '/home/xuejing_liu/yuki/MattNet/cache/word_embedding/vocabulary_72700.txt'
        self.UNK_IDENTIFIER = '<unk>'  # <unk> is the word used to identify unknown words
        self.num_vocab = 72704
        self.embed_dim = 300
        self.embedmat_path = '/home/xuejing_liu/yuki/MattNet/cache/word_embedding/embed_matrix.npy'
        self.embedding_mat = np.load(self.embedmat_path)

        # self.use_bert = 1

        # img_iterators for each split
        self.split_ix = {}
        self.iterators = {}
        for image_id, image in self.Images.items():
            # we use its ref's split (there is assumption that each image only has one split)
            split = self.Refs[image['ref_ids'][0]]['split']
            if split not in self.split_ix:
                self.split_ix[split] = []
                self.iterators[split] = 0
            self.split_ix[split] += [image_id]
        for k, v in self.split_ix.items():
            print('assigned %d images to split %s' %(len(v), k))

    # load vocabulary file
    def load_vocab_dict_from_file(self, dict_file):
        if (sys.version_info > (3, 0)):
            with open(dict_file, encoding='utf-8') as f:
                words = [w.strip() for w in f.readlines()]
        else:
            with io.open(dict_file, encoding='utf-8') as f:
                words = [w.strip() for w in f.readlines()]
        vocab_dict = {words[n]: n for n in range(len(words))}
        return vocab_dict

    # get word location in vocabulary
    def words2vocab_indices(self, words, vocab_dict):
        if isinstance(words, str):
            vocab_indices = [vocab_dict[words] if words in vocab_dict else vocab_dict[self.UNK_IDENTIFIER]]
        else:
            vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[self.UNK_IDENTIFIER])
                             for w in words]
        return vocab_indices

    # get ann category
    def get_ann_category(self, image_id):
        image = self.Images[image_id]
        ann_ids = image['ann_ids']
        vocab_dict = self.load_vocab_dict_from_file(self.vocab_file)
        ann_cats = []

        for ann_id in ann_ids:
            ann = self.Anns[ann_id]
            cat_id = ann['category_id']
            cat = self.ix_to_cat[cat_id]
            cat = cat.split(' ')
            cat = self.words2vocab_indices(cat, vocab_dict)
            ann_cats += [cat]
        return ann_cats

    # get category for each sentence
    def get_sent_category(self, image_id):
        image = self.Images[image_id]
        ref_ids = image['ref_ids']
        vocab_dict = self.load_vocab_dict_from_file(self.vocab_file)
        sub_wds = []
        obj_wds = []
        loc_wds = []
        rel_loc_wds = []
        for ref_id in ref_ids:
            ref = self.Refs[ref_id]
            sent_ids = ref['sent_ids']
            for sent_id in sent_ids:
                att_wds = self.sub_obj_wds[str(sent_id)]
                sub_wd = att_wds['r1']
                sub_wd = self.words2vocab_indices(sub_wd, vocab_dict)
                obj_wd = att_wds['r6']
                obj_wd = self.words2vocab_indices(obj_wd, vocab_dict)
                sub_wds += [sub_wd]
                obj_wds += [obj_wd]
        none_idx = self.words2vocab_indices('none', vocab_dict)


        sent_att = {}
        sent_att['sub_wds'] = sub_wds
        sent_att['obj_wds'] = obj_wds
        return sent_att, none_idx

    def prepare_mrcn(self, head_feats_dir, args):
        """
        Arguments:
          head_feats_dir: cache/feats/dataset_splitBy/net_imdb_tag, containing all image conv_net feats
          args: imdb_name, net_name, iters, tag
        """
        self.head_feats_dir = head_feats_dir
        self.mrcn = inference_no_imdb.Inference(args)
        assert args.net_name == 'res101'
        self.pool5_dim = 1024
        self.fc7_dim = 2048

    # load different kinds of feats
    def loadFeats(self, Feats):
        # Feats = {feats_name: feats_path}
        self.feats = {}
        self.feat_dim = None
        for feats_name, feats_path in Feats.items():
            if osp.isfile(feats_path):
                self.feats[feats_name] = h5py.File(feats_path, 'r')
                self.feat_dim = self.feats[feats_name]['fc7'].shape[1]
                assert self.feat_dim == self.fc7_dim
                print('FeatLoader loading [%s] from %s [feat_dim %s]' %(feats_name, feats_path, self.feat_dim))

    # shuffle split
    def shuffle(self, split):
        random.shuffle(self.split_ix[split])

    # reset iterator
    def resetIterator(self, split):
        self.iterators[split]=0

    # expand list by seq per ref, i.e., [a,b], 3 -> [aaabbb]
    def expand_list(self, L, n):
        out = []
        for l in L:
            out += [l] * n
        return out

    def image_to_head(self, image_id):
        """Returns
        head: float32 (1, 1024, H, W)
        im_info: float32 [[im_h, im_w, im_scale]]
        """
        feats_h5 = osp.join(self.head_feats_dir, str(image_id)+'.h5')
        feats = h5py.File(feats_h5, 'r')
        head, im_info = feats['head'], feats['im_info']
        return np.array(head), np.array(im_info)

    def fetch_neighbour_ids(self, ann_id):
        """
        For a given ann_id, we return
        - st_ann_ids: same-type neighbouring ann_ids (not include itself)
        - dt_ann_ids: different-type neighbouring ann_ids
        Ordered by distance to the input ann_id
        """
        ann = self.Anns[ann_id]
        x,y,w,h = ann['box']
        rx, ry = x+w/2, y+h/2

        @functools.cmp_to_key
        def compare(ann_id0, ann_id1):
            x,y,w,h = self.Anns[ann_id0]['box']
            ax0, ay0 = x+w/2, y+h/2
            x,y,w,h = self.Anns[ann_id1]['box']
            ax1, ay1 = x+w/2, y+h/2
            # closer to farmer
            if (rx-ax0)**2+(ry-ay0)**2 <= (rx-ax1)**2+(ry-ay1)**2:
                return -1
            else:
                return 1

        image = self.Images[ann['image_id']]

        ann_ids = list(image['ann_ids'])
        ann_ids = sorted(ann_ids, key=compare)

        st_ann_ids, dt_ann_ids = [], []
        for ann_id_else in ann_ids:
            if ann_id_else != ann_id:
                if self.Anns[ann_id_else]['category_id'] == ann['category_id']:
                    st_ann_ids += [ann_id_else]
                else:
                    dt_ann_ids +=[ann_id_else]
        return st_ann_ids, dt_ann_ids

    def fetch_grid_feats(self, boxes, net_conv, im_info):
        """returns -pool5 (n, 1024, 7, 7) -fc7 (n, 2048, 7, 7)"""
        pool5, fc7 = self.mrcn.box_to_spatial_fc7(net_conv, im_info, boxes)
        return pool5, fc7

    def compute_lfeats(self, ann_ids):
        # return ndarray float32 (#ann_ids, 5)
        lfeats = np.empty((len(ann_ids), 5), dtype=np.float32)
        for ix, ann_id in enumerate(ann_ids):
            ann = self.Anns[ann_id]
            image = self.Images[ann['image_id']]
            x, y ,w, h = ann['box']
            ih, iw = image['height'], image['width']
            lfeats[ix] = np.array([x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)],np.float32)
        return lfeats

    def compute_dif_lfeats(self, ann_ids, topK=5):
        # return ndarray float32 (#ann_ids, 5*topK)
        dif_lfeats = np.zeros((len(ann_ids), 5*topK), dtype=np.float32)
        for i, ann_id in enumerate(ann_ids):
            # reference box
            rbox = self.Anns[ann_id]['box']
            rcx,rcy,rw,rh = rbox[0]+rbox[2]/2,rbox[1]+rbox[3]/2,rbox[2],rbox[3]
            st_ann_ids, _ =self.fetch_neighbour_ids(ann_id)
            # candidate box
            for j, cand_ann_id in enumerate(st_ann_ids[:topK]):
                cbox = self.Anns[cand_ann_id]['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                dif_lfeats[i, j*5:(j+1)*5] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
        return dif_lfeats


    def fetch_attribute_label(self, ref_ids):
        """Return
    - labels    : Variable float (N, num_atts)
    - select_ixs: Variable long (n, )
    """
        labels = np.zeros((len(ref_ids), self.num_atts))
        select_ixs = []
        for i, ref_id in enumerate(ref_ids):
            # pdb.set_trace()
            ref = self.Refs[ref_id]
            if len(ref['att_wds']) > 0:
                select_ixs += [i]
                for wd in ref['att_wds']:
                    labels[i, self.att_to_ix[wd]] = 1

        return Variable(torch.from_numpy(labels).float().cuda()), Variable(torch.LongTensor(select_ixs).cuda())

    def fetch_cxt_feats(self, ann_ids, opt, obj_sim):
            """
            Return
            - cxt_feats : ndarray (#ann_ids, fc7_dim)
            - cxt_lfeats: ndarray (#ann_ids, 5)
            - dist: ndarray (#ann_ids, ann_ids, 1)
            Note we only use neighbouring "different" (+ "same") objects for computing context objects, zeros padded.
            """
            cxt_feats = np.zeros((len(ann_ids), self.fc7_dim), dtype=np.float32)
            cxt_lfeats = np.zeros((len(ann_ids), 5), dtype=np.float32)
            cxt_lfeats_rel = np.zeros((len(ann_ids), len(ann_ids), 5), dtype=np.float32)

            for i, ann_id in enumerate(ann_ids):
                # reference box
                rbox = self.Anns[ann_id]['box']
                rcx, rcy, rw, rh = rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
                for j, cand_ann_id in enumerate(ann_ids):
                    cand_ann = self.Anns[cand_ann_id]
                    cxt_feats[i, :] = self.feats['ann']['fc7'][cand_ann['h5_id'], :]
                    image = self.Images[cand_ann['image_id']]
                    cbox = cand_ann['box']
                    cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                    ih, iw = image['height'], image['width']
                    cxt_lfeats[i, :] = np.array([cx1/iw, cy1/ih, (cx1+cw-1)/iw, (cy1+ch-1)/ih, cw*ch/(iw*ih)],np.float32)
                    cxt_lfeats_rel[i,j,:] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
            return cxt_feats, cxt_lfeats, cxt_lfeats_rel


    def extract_ann_features(self, image_id, opt, obj_sim):
        """Get features for all ann_ids in an image"""
        image = self.Images[image_id]
        ann_ids = image['ann_ids']

        # fetch image features
        head, im_info = self.image_to_head(image_id)
        head = Variable(torch.from_numpy(head).cuda())

        # fetch ann features
        ann_boxes = xywh_to_xyxy(np.vstack([self.Anns[ann_id]['box'] for ann_id in ann_ids]))
        ann_pool5, ann_fc7 = self.fetch_grid_feats(ann_boxes, head, im_info)

        # absolute location features
        lfeats = self.compute_lfeats(ann_ids)
        lfeats = Variable(torch.from_numpy(lfeats).cuda())

        # relative location features
        dif_lfeats = self.compute_dif_lfeats(ann_ids)
        dif_lfeats = Variable(torch.from_numpy(dif_lfeats).cuda())

        # fetch context_fc7 and context_lfeats
        cxt_fc7, cxt_lfeats, cxt_lfeats_rel = self.fetch_cxt_feats(ann_ids, opt, obj_sim)
        cxt_fc7 = Variable(torch.from_numpy(cxt_fc7).cuda())
        cxt_lfeats = Variable(torch.from_numpy(cxt_lfeats).cuda())
        cxt_lfeats_rel = Variable(torch.from_numpy(cxt_lfeats_rel).cuda())
        # dist = Variable(torch.from_numpy(dist).cuda())

        return ann_fc7, ann_pool5, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, cxt_lfeats_rel


    def compute_rel_lfeats(self, sub_ann_id, obj_ann_id):
        if sub_ann_id == -1 or obj_ann_id == -1:
            rel_lfeats = torch.zeros(5)
        else:
            rbox = self.Anns[sub_ann_id]['box']
            rcx, rcy, rw, rh = rbox[0] + rbox[2] / 2, rbox[1] + rbox[3] / 2, rbox[2], rbox[3]
            cbox = self.Anns[obj_ann_id]['box']
            cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
            rel_lfeats = np.array(
                [(cx1 - rcx) / rw, (cy1 - rcy) / rh, (cx1 + cw - rcx) / rw, (cy1 + ch - rcy) / rh, cw * ch / (rw * rh)])
            rel_lfeats = torch.Tensor(rel_lfeats)
        return rel_lfeats

    # get batch of data
    def getBatch(self, split, opt):
        # options
        # 一张图像作为一个batch
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        wrapped = False
        TopK = opt['num_cxt']

        ri = self.iterators[split]
        ri_next = ri+1
        if ri_next > max_index:
            ri_next = 0
            wrapped = True
        self.iterators[split] = ri_next
        image_id = split_ix[ri]

        ann_ids = self.Images[image_id]['ann_ids']
        ann_num = len(ann_ids)
        ref_ids = self.Images[image_id]['ref_ids']

        img_ref_ids = []
        img_sent_ids = []
        gd_ixs = []
        gd_boxes = []
        for ref_id in ref_ids:
            ref = self.Refs[ref_id]
            for sent_id in ref['sent_ids']:
                img_ref_ids += [ref_id]
                img_sent_ids += [sent_id]
                gd_ixs += [ann_ids.index(ref['ann_id'])]
                gd_boxes += [ref['box']]
        img_sent_num = len(img_sent_ids)

        sub_sim = torch.Tensor(self.similarity[str(image_id)]['sub_sim']).cuda().squeeze(2)
        obj_sim = torch.Tensor(self.similarity[str(image_id)]['obj_sim']).cuda().squeeze(2)
        sub_emb = torch.Tensor(self.similarity[str(image_id)]['sub_emb']).cuda()
        obj_emb = torch.Tensor(self.similarity[str(image_id)]['obj_emb']).cuda()

        ann_fc7, ann_pool5, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, cxt_lfeats_rel = self.extract_ann_features(image_id, opt, obj_sim)

        pool5 = ann_pool5.unsqueeze(0).expand(img_sent_num, ann_num, self.pool5_dim, 7, 7)
        pool5.detach()
        fc7 = ann_fc7.unsqueeze(0).expand(img_sent_num, ann_num, self.fc7_dim, 7, 7)
        fc7.detach()
        lfeats = lfeats.unsqueeze(0).expand(img_sent_num, ann_num, 5)
        lfeats.detach()
        dif_lfeats = dif_lfeats.unsqueeze(0).expand(img_sent_num, ann_num, TopK*5)
        dif_lfeats.detach()
        cxt_fc7.detach()
        cxt_lfeats.detach()
        cxt_lfeats_rel.detach()
        # dist.detach() #(ann_ids, ann_ids, 1)
        sub_emb.detach()
        obj_emb.detach()
        sub_sim.detach()
        obj_sim.detach()

        att_labels, select_ixs = self.fetch_attribute_label(img_ref_ids)

        # if self.use_bert == 1:
        #     labels = np.vstack([self.fetch_seq_bert(sent_id) for sent_id in img_sent_ids])
        # else:
        #     labels = np.vstack([self.fetch_seq(sent_id) for sent_id in img_sent_ids])

        labels = np.vstack([self.fetch_seq(sent_id) for sent_id in img_sent_ids])

        labels = Variable(torch.from_numpy(labels).long().cuda())
        max_len = (labels!=0).sum(1).max().data[0]
        labels = labels[:, :max_len]

        start_words = np.ones([labels.size(0), 1], dtype=int)*(self.word_to_ix['<BOS>'])
        start_words = Variable(torch.from_numpy(start_words).long().cuda())
        enc_labels = labels.clone()
        enc_labels = torch.cat([start_words, enc_labels], 1)

        zero_pad = np.zeros([labels.size(0), 1], dtype=int)
        zero_pad = Variable(torch.from_numpy(zero_pad).long().cuda())
        dec_labels = labels.clone()
        dec_labels = torch.cat([dec_labels, zero_pad], 1)

        # pdb.set_trace()
        data = {}
        data['labels'] = labels
        data['enc_labels'] = enc_labels
        data['dec_labels'] = dec_labels
        data['ref_ids'] = ref_ids
        data['sent_ids'] = img_sent_ids
        data['gd_ixs'] = gd_ixs
        data['gd_boxes'] = gd_boxes
        data['sim'] = {'sub_sim': sub_sim, 'obj_sim': obj_sim, 'sub_emb': sub_emb, 'obj_emb': obj_emb}
        data['Feats'] = {'fc7': fc7, 'pool5': pool5, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                          'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats, 'cxt_lfeats_rel': cxt_lfeats_rel}
        data['att_labels'] = att_labels
        data['select_ixs'] = select_ixs
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}
        return data

    def get_attribute_weights(self, scale = 10):
        # weights = \lamda * 1/sqrt(cnt)
        cnts = [self.att_to_cnt[self.ix_to_att[ix]] for ix in range(self.num_atts)]
        cnts = np.array(cnts)
        weights = 1 / cnts ** 0.5
        weights = (weights-np.min(weights))/(np.max(weights)-np.min(weights))
        weights = weights * (scale - 1) + 1
        return torch.from_numpy(weights).float()

    def decode_attribute_label(self, scores):
        """- scores: Variable (cuda) (n, num_atts) after sigmoid range [0, 1]
           - labels:list of [[att, sc], [att, sc], ...
        """
        scores = scores.data.cpu().numpy()
        N = scores.shape[0]
        labels = []
        for i in range(N):
            label = []
            score = scores[i]
            for j, sc in enumerate(list(score)):
                label += [(self.ix_to_att[j], sc)]
                labels.append(label)
        return labels

    def getTestBatch(self, split, opt):
        TopK = opt['num_cxt']
        wrapped = False
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        ri = self.iterators[split]
        ri_next = ri + 1
        if ri_next > max_index:
            ri_next = 0
            wrapped = True
        self.iterators[split] = ri_next
        image_id = split_ix[ri]
        image = self.Images[image_id]
        ann_ids = image['ann_ids']

        sent_ids = []
        gd_ixs = []
        gd_boxes = []
        att_refs = []
        for ref_id in image['ref_ids']:
            ref = self.Refs[ref_id]
            for sent_id in ref['sent_ids']:
                sent_ids += [sent_id]
                gd_ixs += [ann_ids.index(ref['ann_id'])]
                gd_boxes += [ref['box']]
                att_refs += [ref_id]

        # get ann category
        ann_cats = self.get_ann_category(image_id)
        # get subject, object, location words for each sentence
        sent_att, none_idx = self.get_sent_category(image_id)
        # compute similarity
        self.com_sim = com_simV2.ComSim(self.embedding_mat)
        # 输出的是tensor
        sub_sim, sub_emb = self.com_sim.cal_sim(ann_cats, sent_att['sub_wds'], none_idx) # (sent_num, ann_num)
        obj_sim, obj_emb = self.com_sim.cal_sim(ann_cats, sent_att['obj_wds'], none_idx) # (sent_num, ann_num)

        ann_fc7, ann_pool5, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, cxt_lfeats_rel = self.extract_ann_features(image_id, opt,
                                                                                                      obj_sim)

        pool5 = ann_pool5.unsqueeze(0)
        pool5.detach()
        fc7 = ann_fc7.unsqueeze(0)
        fc7.detach()
        lfeats = lfeats.unsqueeze(0)
        lfeats.detach()
        dif_lfeats = dif_lfeats.unsqueeze(0)
        dif_lfeats.detach()
        cxt_fc7.detach()
        cxt_lfeats.detach()
        cxt_lfeats_rel.detach()
        sub_emb.detach()
        obj_emb.detach()
        sub_sim.detach()
        obj_sim.detach()
        # dist.detach()

        labels = np.vstack([self.fetch_seq(sent_id) for sent_id in sent_ids])
        labels = Variable(torch.from_numpy(labels).long().cuda())
        max_len = (labels!=0).sum(1).max().data[0]
        labels = labels[:, :max_len]

        start_words = np.ones([labels.size(0), 1], dtype=int)*(self.word_to_ix['<BOS>'])
        start_words = Variable(torch.from_numpy(start_words).long().cuda())
        enc_labels = labels.clone()
        enc_labels = torch.cat([start_words, enc_labels], 1)

        zero_pad = np.zeros([labels.size(0), 1], dtype=int)
        zero_pad = Variable(torch.from_numpy(zero_pad).long().cuda())
        dec_labels = labels.clone()
        dec_labels = torch.cat([dec_labels, zero_pad], 1)

        att_labels, select_ixs = self.fetch_attribute_label(att_refs)


        data = {}
        data['image_id'] = image_id
        data['ann_ids'] = ann_ids
        data['sent_ids'] = sent_ids
        data['gd_ixs'] = gd_ixs
        data['gd_boxes'] = gd_boxes
        data['sim'] = {'sub_sim': sub_sim, 'obj_sim': obj_sim, 'sub_emb': sub_emb, 'obj_emb': obj_emb}
        data['Feats'] = {'fc7': fc7, 'pool5': pool5, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                          'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats, 'cxt_lfeats_rel': cxt_lfeats_rel}
        data['labels'] = labels
        data['enc_labels'] = enc_labels
        data['dec_labels'] = dec_labels
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}
        data['att_labels'] = att_labels
        data['select_ixs'] = select_ixs

        return data



def main(args):
    start_time = time.time()
    RootPath = '/home/xuejing_liu/yuki/MattNet'
    opt = vars(args)
    # initialize
    opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']
    # output/refcoco_unc
    checkpoint_dir = osp.join(opt['checkpoint_path'], opt['dataset_splitBy'], opt['exp_id'])
    if not osp.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 使用TensroboardX可视化
    writer = SummaryWriter(osp.join(checkpoint_dir, 'log'))

    # set random seed
    # seed()  用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed()
    # 值，则每次生成的随机数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
    torch.manual_seed(opt['seed'])
    random.seed(opt['seed'])

    # set up loader
    data_json = osp.join(RootPath, 'cache/prepro', opt['dataset_splitBy'], 'data.json')
    data_h5 = osp.join(RootPath, 'cache/prepro', opt['dataset_splitBy'], 'data.h5')
    sub_obj_wds = osp.join(RootPath, 'cache/sub_obj_wds', opt['dataset_splitBy'], 'sub_obj.json')
    similarity = osp.join(RootPath, 'cache/similarity', opt['dataset_splitBy'], 'similarity.json')
    loader = DataLoader(data_h5=data_h5, data_json=data_json, sub_obj_wds=sub_obj_wds, similarity=similarity, opt=opt)

    # prepare feats
    feats_dir = '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
    head_feats_dir = osp.join(RootPath, 'cache/feats/', opt['dataset_splitBy'], 'mrcn', feats_dir)

    loader.prepare_mrcn(head_feats_dir, args)

    ann_feats = osp.join(RootPath, 'cache/feats', opt['dataset_splitBy'], 'mrcn',
                         '%s_%s_%s_ann_feats.h5' % (opt['net_name'], opt['imdb_name'], opt['tag']))
    loader.loadFeats({'ann': ann_feats})

    # set up model
    opt['vocab_size'] = loader.vocab_size
    opt['fc7_dim'] = loader.fc7_dim
    opt['pool5_dim'] = loader.pool5_dim
    opt['num_atts'] = loader.num_atts

    model = KARN(opt)

    infos = {}
    if opt['start_from'] is not None:
        pass
    iter = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_accuracies = infos.get('val_accuracies', [])
    val_loss_history = infos.get('val_loss_history', {})
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    if opt['load_best_score'] == 1:
        best_val_score = infos.get('best_val_score', None)

    att_weights = loader.get_attribute_weights()

    if opt['gpuid'] >= 0:
        model.cuda()

    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt['learning_rate'],
                                 betas=(opt['optim_alpha'], opt['optim_beta']),
                                 eps=opt['optim_epsilon'],
                                 weight_decay=opt['weight_decay'])

    data_time, model_time = 0, 0
    lr = opt['learning_rate']
    best_prediction, best_overall = None, None
    while True:
        model.train()
        optimizer.zero_grad()

        T = {}

        tic = time.time()
        data = loader.getBatch('train', opt)

        labels = data['labels']
        enc_labels = data['enc_labels']
        dec_labels = data['dec_labels']
        Feats = data['Feats']
        sub_sim = data['sim']['sub_sim']
        obj_sim = data['sim']['obj_sim']
        sub_emb = data['sim']['sub_emb']
        obj_emb = data['sim']['obj_emb']
        att_labels, select_ixs = data['att_labels'], data['select_ixs']

        T['data'] = time.time() - tic

        tic = time.time()
        scores, loss, _, _, _, _, _, _, _, _, _, _= model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'], Feats['cxt_fc7'],
                             Feats['cxt_lfeats'],Feats['cxt_lfeats_rel'], labels, enc_labels, dec_labels, sub_sim, obj_sim, sub_emb, obj_emb, att_labels, select_ixs, att_weights)

        loss.backward()
        model_utils.clip_gradient(optimizer, opt['grad_clip'])
        # pdb.set_trace()
        optimizer.step()
        T['model'] = time.time() - tic
        wrapped = data['bounds']['wrapped']

        data_time += T['data']
        model_time += T['model']

        if iter % opt['losses_log_every'] == 0:
            loss_history[iter] = (loss.data[0]).item()
            # pdb.set_trace()
            print('iter[%s](epoch[%s]), train_loss=%.3f, lr=%.2E, data:%.2fs/iter, model:%.2fs/iter' \
                  % (iter, epoch, loss.data[0].item(), lr, data_time / opt['losses_log_every'],
                     model_time / opt['losses_log_every']))
            writer.add_scalar('Train/Loss', loss.data[0].item(), iter)

            data_time, model_time = 0, 0

        if opt['learning_rate_decay_start'] > 0 and iter > opt['learning_rate_decay_start']:
            frac = (iter - opt['learning_rate_decay_start']) / opt['learning_rate_decay_every']
            decay_factor = 0.1 ** frac
            lr = opt['learning_rate'] * decay_factor
            model_utils.set_lr(optimizer, lr)

        if iter % opt['save_checkpoint_every'] == 0 or iter == opt['max_iters']:

            val_loss, acc, predictions = eval_split(loader, model, 'val', opt)
            val_loss_history[iter] = val_loss
            val_result_history[iter] = {'loss': val_loss, 'accuracy': acc}
            val_accuracies += [(iter, acc)]
            print('validation loss: %.2f' % val_loss)
            print('validation acc : %.2f%%\n' % (acc * 100.0))
            writer.add_scalar('Test/Loss', val_loss, iter)
            writer.add_scalar('Test/Acc', acc, iter)

            current_score = acc
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_predictions = predictions
                checkpoint_path = osp.join(checkpoint_dir, opt['id'] + '.pth')
                checkpoint = {}
                checkpoint['model'] = model
                checkpoint['opt'] = opt
                torch.save(checkpoint, checkpoint_path)
                print('model saved to %s' % checkpoint_path)

                # write json report
            infos['iter'] = iter
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['loss_history'] = loss_history
            infos['val_accuracies'] = val_accuracies
            infos['val_loss_history'] = val_loss_history
            infos['best_val_score'] = best_val_score
            infos['best_predictions'] = predictions if best_predictions is None else best_predictions

            infos['opt'] = opt
            infos['val_result_history'] = val_result_history
            # infos['word_to_ix'] = loader.word_to_ix

            with open(osp.join(checkpoint_dir, opt['id'] + '.json'), 'w', encoding="utf8") as io:
                json.dump(infos, io)

        iter += 1
        if wrapped:
            epoch += 1
        if iter > opt['max_iters'] and opt['max_iters'] > 0:
            print(time.time()-start_time)
            break

if __name__ == '__main__':
    args = parse_opt()
    main(args)
