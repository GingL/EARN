from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import json
import numpy as np
import h5py
import time
from pprint import pprint
import argparse
import random

import torch
from torch.autograd import Variable

# model
import _init_paths
from layers.modelV2 import KARN
import evals.eval_easyV2 as eval_utils
from loaders.loader import Loader
from mrcn import inference_no_imdb
from sim_utils import com_simV2
from sim_utils import obj_ann_filter
from sim_utils import sub_ann_filter
import functools
import pdb


# box functions
def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

class DataLoader(Loader):

    def __init__(self, data_json, data_h5, sub_obj_wds, similarity, opt, complex_rel):
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
        self.complex_rel = complex_rel
        self.iterators = 0

        print('%d complex relationsips' %(len(self.complex_rel)))

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
        self.iterators=0

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
        - cxt_lfeats: ndarray (#ann_ids, ann_ids, 5)
        - dist: ndarray (#ann_ids, ann_ids, 1)
        Note we only use neighbouring "different" (+ "same") objects for computing context objects, zeros padded.
        """

        cxt_feats = np.zeros((len(ann_ids), self.fc7_dim), dtype=np.float32)
        cxt_lfeats = np.zeros((len(ann_ids), len(ann_ids), 5), dtype=np.float32)
        dist = np.zeros((len(ann_ids), len(ann_ids), 1), dtype=np.float32)

        for i, ann_id in enumerate(ann_ids):
            # reference box
            rbox = self.Anns[ann_id]['box']
            rcx, rcy, rw, rh = rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
            for j, cand_ann_id in enumerate(ann_ids):
                cand_ann = self.Anns[cand_ann_id]
                # fc7_feats
                # import ipdb
                # ipdb.set_trace()
                cxt_feats[i, :] = self.feats['ann']['fc7'][cand_ann['h5_id'], :]
                cbox = cand_ann['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                ccx, ccy = cbox[0]+cbox[2]/2, cbox[1]+cbox[3]/2
                dist[i,j,:] = np.array(abs(rcx-ccx)+abs(rcy-ccy))
                cxt_lfeats[i,j,:] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
        return cxt_feats, cxt_lfeats, dist

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
        cxt_fc7, cxt_lfeats, dist = self.fetch_cxt_feats(ann_ids, opt, obj_sim)
        cxt_fc7 = Variable(torch.from_numpy(cxt_fc7).cuda())
        cxt_lfeats = Variable(torch.from_numpy(cxt_lfeats).cuda())
        dist = Variable(torch.from_numpy(dist).cuda())

        return ann_fc7, ann_pool5, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, dist


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

        ann_fc7, ann_pool5, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, dist = self.extract_ann_features(image_id, opt, obj_sim)

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
        dist.detach() #(ann_ids, ann_ids, 1)
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
                          'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats, 'dist': dist}
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
        complex_rel = self.complex_rel
        max_index = len(complex_rel) - 1
        ri = self.iterators
        ri_next = ri + 1
        if ri_next > max_index:
            ri_next = 0
            wrapped = True
        self.iterators = ri_next
        rel = complex_rel[ri]
        ref_id  = rel['ref_id']
        sent_id = rel['sent_id']
        sent = rel['sent']

        ref = self.Refs[ref_id]
        gd_boxes = ref['box']
        image_id = ref['image_id']
        image = self.Images[image_id]
        ann_ids = image['ann_ids']
        gd_ixs = ann_ids.index(ref['ann_id'])

        # get ann category
        ann_cats = self.get_ann_category(image_id)
        # get subject, object, location words for each sentence
        sent_att, none_idx = self.get_sent_category(image_id)
        # compute similarity
        self.com_sim = com_simV2.ComSim(self.embedding_mat)
        # 输出的是tensor
        sub_sim, sub_emb = self.com_sim.cal_sim(ann_cats, sent_att['sub_wds'], none_idx) # (sent_num, ann_num)
        obj_sim, obj_emb = self.com_sim.cal_sim(ann_cats, sent_att['obj_wds'], none_idx) # (sent_num, ann_num)

        ann_fc7, ann_pool5, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, dist = self.extract_ann_features(image_id, opt,
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
        sub_emb.detach()
        obj_emb.detach()
        sub_sim.detach()
        obj_sim.detach()
        dist.detach()

        labels = self.fetch_seq(sent_id)
        labels = Variable(torch.from_numpy(labels).long().cuda())
        labels = labels.unsqueeze(0)
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

        att_labels, select_ixs = self.fetch_attribute_label([ref_id])


        data = {}
        data['image_id'] = image_id
        data['ann_ids'] = ann_ids
        data['sent_ids'] = [sent_id]
        data['gd_ixs'] = [gd_ixs]
        data['gd_boxes'] = [gd_boxes]
        data['sim'] = {'sub_sim': sub_sim, 'obj_sim': obj_sim, 'sub_emb': sub_emb, 'obj_emb': obj_emb}
        data['Feats'] = {'fc7': fc7, 'pool5': pool5, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                          'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats, 'dist': dist}
        data['labels'] = labels
        data['enc_labels'] = enc_labels
        data['dec_labels'] = dec_labels
        data['bounds'] = {'it_pos_now': self.iterators, 'it_max': max_index, 'wrapped': wrapped}
        data['att_labels'] = att_labels
        data['select_ixs'] = select_ixs

        return data


def load_model(checkpoint_path, opt):
    tic = time.time()
    model = KARN(opt)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'].state_dict())
    model.eval()
    model.cuda()
    print('model loaded in %.2f seconds' % (time.time() - tic))
    return model

def evaluate(params):
    RootPath = '/home/xuejing_liu/yuki/MattNet'

    # load mode info
    model_prefix = osp.join('output', params['dataset_splitBy'], params['id'], 'mrcn_cmr_with_st')
    infos = json.load(open(model_prefix + '.json'))
    model_opt = infos['opt']
    model_path = model_prefix + '.pth'
    model = load_model(model_path, model_opt)

    # set up loader
    complex_rel_path = osp.join(RootPath, 'cache/prepro', params['dataset_splitBy'], 'complex_rel_eval.pth')
    complex_rel = torch.load(complex_rel_path)
    data_json = osp.join(RootPath, 'cache/prepro', params['dataset_splitBy'], 'data.json')
    data_h5 = osp.join(RootPath, 'cache/prepro', params['dataset_splitBy'], 'data.h5')
    sub_obj_wds = osp.join(RootPath, 'cache/sub_obj_wds', model_opt['dataset_splitBy'], 'sub_obj.json')
    similarity = osp.join(RootPath, 'cache/similarity', model_opt['dataset_splitBy'], 'similarity.json')
    loader = DataLoader(data_h5=data_h5, data_json=data_json, sub_obj_wds=sub_obj_wds, similarity=similarity, 
                        opt=model_opt, complex_rel=complex_rel)

    # load refer
    data_root = '/home/xuejing_liu/yuki/MattNet/data'
    dataset = params['dataset']
    splitBy = params['splitBy']
    sys.path.insert(0, '/home/xuejing_liu/yuki/MattNet/pyutils/refer')
    from refer import REFER
    refer = REFER(data_root, dataset, splitBy)



    # loader's feats
    feats_dir = '%s_%s_%s' % (model_opt['net_name'], model_opt['imdb_name'], model_opt['tag'])
    # imdb_name = coco_minus_refer
    args.imdb_name = model_opt['imdb_name']
    # net_name = res101
    args.net_name = model_opt['net_name']
    # tag = notime
    args.tag = model_opt['tag']
    args.iters = model_opt['iters']
    loader.prepare_mrcn(head_feats_dir=osp.join(RootPath, 'cache/feats/', model_opt['dataset_splitBy'], 'mrcn', feats_dir),
                        args=args)
    ann_feats = osp.join(RootPath, 'cache/feats', model_opt['dataset_splitBy'], 'mrcn',
                         '%s_%s_%s_ann_feats.h5' % (model_opt['net_name'], model_opt['imdb_name'], model_opt['tag']))
    # load ann features
    loader.loadFeats({'ann': ann_feats})

    # check model_info and params
    assert model_opt['dataset'] == params['dataset']
    assert model_opt['splitBy'] == params['splitBy']

    # evaluate on the split,
    # predictions = [{sent_id, sent, gd_ann_id, pred_ann_id, pred_score, sub_attn, loc_attn, weights}]
    split = params['split']
    model_opt['num_sents'] = params['num_sents']
    model_opt['verbose'] = params['verbose']

    val_loss, acc, predictions = eval_utils.eval_split(loader, model, split, model_opt)
    # 可视化
    # draw(refer, loader, predictions, params)

    print('Comprehension on %s\'s %s (%s sents) is %.2f%%' % \
          (params['dataset_splitBy'], params['split'], len(predictions), acc * 100.))
    # print('attribute precision : %.2f%%' % (overall['precision'] * 100.0))
    # print('attribute recall    : %.2f%%' % (overall['recall'] * 100.0))
    # print('attribute f1        : %.2f%%' % (overall['f1'] * 100.0))

    # save
    out_dir = osp.join('results', params['dataset_splitBy'], 'easy')
    if not osp.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = osp.join(out_dir, params['id'] + '_' + params['split'] + '.json')
    with open(out_file, 'w') as of:
        # json.dump({'predictions': predictions, 'acc': acc, 'overall': overall}, of)
        json.dump({'predictions': predictions, 'acc': acc}, of)

    # write to results.txt
    f = open('experiments/results.txt', 'a')
    f.write('complex relationship: [%s]: [%s][%s], id[%s]\'s acc is %.2f%%\n' % \
            (params['id'], params['dataset_splitBy'], params['split'], params['id'], acc * 100.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='refcoco',
                        help='dataset name: refclef, refcoco, refcoco+, refcocog')
    parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
    parser.add_argument('--split', type=str, default='val', help='split: testAB or val, etc')
    parser.add_argument('--id', type=str, default='0', help='model id name')
    parser.add_argument('--num_sents', type=int, default=-1,
                        help='how many sentences to use when periodically evaluating the loss? (-1=all)')
    parser.add_argument('--verbose', type=int, default=1, help='if we want to print the testing progress')
    args = parser.parse_args()
    params = vars(args)

    # make other options
    params['dataset_splitBy'] = params['dataset'] + '_' + params['splitBy']
    evaluate(params)


