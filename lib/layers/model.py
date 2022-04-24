from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from layers.lan_enc import RNNEncoder, PhraseAttention
from layers.lan_dec import RNNDncoder, SubjectDecoder, LocationDecoder, RelationDecoder
from layers.vis_enc import LocationEncoder, SubjectEncoder
from layers.reconstruct_loss import AttributeReconstructLoss, LangLangReconstructLoss, VisualLangReconstructLoss, ReconstructionLoss


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

    def forward(self, cxt_feats, cxt_lfeats, obj_attn):
        # cxt_feats.shape = (ann_num,2048), cxt_lfeats=(ann_num,ann_num,5), obj_attn=(sent_num,ann_num),
        # dist=(ann_num,ann_num,1), wo_obj_idx.shape=(sent_num)

        sent_num = obj_attn.size(0)
        ann_num = cxt_feats.size(0)
        batch = sent_num * ann_num

        # cxt_feats
        cxt_feats = cxt_feats.unsqueeze(0).expand(sent_num, ann_num, self.fc7_dim)  # cxt_feats.shape = (sent_num，ann_num,2048)
        obj_attn = obj_attn.unsqueeze(1)  # obj_attn=(sent_num, 1, ann_num)
        cxt_feats = torch.bmm(obj_attn, cxt_feats)  # cxt_feats_fuse.shape = (sent_num，1,2048)
        cxt_feats = self.fc7_normalizer(cxt_feats.contiguous().view(sent_num, -1))
        cxt_feats = cxt_feats.unsqueeze(1).expand(sent_num, ann_num, self.fc7_dim)

        cxt_lfeats = cxt_lfeats.unsqueeze(0).expand(sent_num, ann_num, ann_num, 5)
        cxt_lfeats = cxt_lfeats.contiguous().view(batch, ann_num, 5)
        obj_attn = obj_attn.unsqueeze(1).expand(sent_num, ann_num, 1, ann_num)
        obj_attn = obj_attn.contiguous().view(batch, 1, ann_num)
        cxt_lfeats = torch.bmm(obj_attn, cxt_lfeats) # (batch, 1, 5)
        cxt_lfeats = self.lfeat_normalizer(cxt_lfeats.squeeze(1))
        cxt_lfeats = cxt_lfeats.view(sent_num, ann_num, -1)

        cxt_feats_fuse = torch.cat([cxt_feats, cxt_lfeats], 2)

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


class EARN(nn.Module):
    def __init__(self, opt):
        super(EARN, self).__init__()
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
        self.rel_score = Score(self.fc7_dim+5, opt['word_vec_size'],
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
            nn.Linear(self.fc7_dim + self.pool5_dim + 25 + 5 + self.fc7_dim + 5, opt['jemb_dim']))


    def forward(self, pool5, fc7, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, labels, enc_labels, dec_labels,
                sub_sim, obj_sim, sub_emb, obj_emb, att_labels, select_ixs, att_weights):

        sent_num = pool5.size(0)
        ann_num =  pool5.size(1)

        # language feats
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
        # subject filter
        sub_idx = sub_sim.gt(self.filter_thr)
        all_filterd_idx = (sub_idx.sum(1).eq(0))  # (sent_num)
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
        rel_feats = self.rel_encoder(cxt_fc7, cxt_lfeats, obj_attn)  # (sent_num, ann_num, 2048+5) (sent_num, ann_num)
        # dist = 100 / (dist + 100)

        sub_ann_attn = self.sub_score(sub_feats, sub_phrase_emb)  # (sent_num, ann_num, 1)
        loc_ann_attn = self.loc_score(loc_feats, loc_phrase_emb)  # (sent_num, ann_num, 1)
        rel_ann_attn = self.rel_score(rel_feats, rel_phrase_emb)  # (sent_num, ann_num, 1)

        weights_expand = weights.unsqueeze(1).expand(sent_num, ann_num, 3)
        total_ann_score = (weights_expand * torch.cat([sub_ann_attn, loc_ann_attn, rel_ann_attn], 2)).sum(2)  # (sent_num, ann_num)


        sub_phrase_recons = self.sub_decoder(sub_feats, total_ann_score)  # (sent_num, 512)
        loc_phrase_recons = self.loc_decoder(loc_feats, total_ann_score)  # (sent_num, 512)
        rel_phrase_recons = self.rel_decoder(rel_feats, total_ann_score)  # (sent_num, 512)

        loss = 0
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
