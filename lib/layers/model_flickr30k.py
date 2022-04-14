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
from layers.lan_dec import RNNDncoder, SubjectDecoder, RelationDecoder
from layers.reconstruct_loss import ReconstructionLoss

class SubjectEncoder(nn.Module):
    def __init__(self, opt):
        super(SubjectEncoder, self).__init__()
        self.word_vec_size = opt['word_vec_size']
        self.jemb_dim = opt['jemb_dim']
        self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']

        self.pool5_normalizer = Normalize_Scale(opt['pool5_dim'], opt['visual_init_norm'])
        self.fc7_normalizer = Normalize_Scale(opt['fc7_dim'], opt['visual_init_norm'])

    def forward(self, pool5, fc7):
        # fc7.shape = (sent_num,ann_num,2048,7,7), pool5.shape=(sent_num,ann_num,1024,7,7)
        # sent_num, ann_num = pool5.size(0), pool5.size(1)
        # batch = sent_num * ann_num

        # pool5 = pool5.contiguous().view(batch, self.pool5_dim, -1)  # (sent_num * ann_num, 1024, 49)
        # pool5 = pool5.transpose(1, 2).contiguous().view(-1, self.pool5_dim)  # (sent_num * ann_num * 49, 1024)
        # pool5 = self.pool5_normalizer(pool5)  # (sent_num * ann_num * 49, 1024)
        # pool5 = pool5.view(sent_num, ann_num, 49, -1).transpose(2, 3).contiguous().mean(3)  # (sent_num, ann_num, 1024)

        # fc7 = fc7.contiguous().view(batch, self.fc7_dim, -1)  # (sent_num * ann_num, 2048, 49)
        # fc7 = fc7.transpose(1, 2).contiguous().view(-1, self.fc7_dim)  # (sent_num * ann_num * 49, 2048)
        # fc7 = self.fc7_normalizer(fc7)  # (sent_num * ann_num * 49, 2048)
        # fc7 = fc7.view(sent_num, ann_num, 49, -1).transpose(2, 3).contiguous().mean(3)  # (sent_num, ann_num, 2048)

        avg_att_feats = torch.cat([pool5, fc7], 2)  # (sent_num, ann_num, 2048+1024)

        return avg_att_feats

class AttributeReconstructLoss(nn.Module):
    def __init__(self, opt):
        super(AttributeReconstructLoss, self).__init__()
        self.att_dropout = nn.Dropout(opt['visual_drop_out'])
        self.att_fc = nn.Linear(opt['fc7_dim']+opt['pool5_dim'], opt['num_atts'])


    def forward(self, attribute_feats, total_ann_score, att_labels, select_ixs, att_weights):
        """attribute_feats.shape = (sent_num, ann_num, 512), total_ann_score.shape = (sent_num, ann_num)"""
        total_ann_score = total_ann_score.unsqueeze(1)  # (sent_num, 1, ann_num)
        att_feats_fuse = torch.bmm(total_ann_score, attribute_feats)  # (sent_num, 1, 2048+1024)
        att_feats_fuse = att_feats_fuse.squeeze(1)  # (sent_num, 512)
        att_feats_fuse = self.att_dropout(att_feats_fuse)  # dropout
        att_scores = self.att_fc(att_feats_fuse)  # (sent_num, num_atts)
        att_loss = nn.BCEWithLogitsLoss(att_weights.cuda())(att_scores*select_ixs.float(),
                                                    att_labels*select_ixs.float())   

        # if len(select_ixs) == 0:
        #     att_loss = 0
        # else:
        #     att_loss = nn.BCEWithLogitsLoss(att_weights.cuda())(att_scores.index_select(0, select_ixs),
        #                                              att_labels.index_select(0, select_ixs))
        return att_scores, att_loss

class LangLangReconstructLoss(nn.Module):
    def __init__(self, opt):
        super(LangLangReconstructLoss, self).__init__()

        self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']

        self.variable_lengths = opt['variable_lengths'] > 0
        self.vocab_size = opt['vocab_size']
        self.word_embedding_size = opt['word_embedding_size']
        self.word_vec_size = opt['word_vec_size']
        self.hidden_size = opt['rnn_hidden_size']
        self.bidirectional = opt['decode_bidirectional'] > 0
        self.input_dropout_p = opt['word_drop_out']
        self.dropout_p = opt['rnn_drop_out']
        self.n_layers = opt['rnn_num_layers']
        self.rnn_type = opt['rnn_type']
        self.variable_lengths = opt['variable_lengths'] > 0


        self.embedding = nn.Embedding(self.vocab_size, self.word_embedding_size)
        self.input_dropout = nn.Dropout(self.input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(self.word_embedding_size, self.word_vec_size), nn.ReLU())
        self.rnn_type = self.rnn_type
        self.rnn = getattr(nn, self.rnn_type.upper())(self.word_vec_size * 2, self.hidden_size, self.n_layers,
                                                      batch_first=True, bidirectional=self.bidirectional,
                                                      dropout=self.dropout_p)
        self.num_dirs = 2 if self.bidirectional else 1

        self.slr_mlp = nn.Sequential(nn.Linear(self.word_vec_size * 2, self.word_vec_size),
                                     nn.ReLU())
        # self.slr_mlp = nn.Sequential(nn.Linear(self.pool5_dim+self.fc7_dim+25+5+self.fc7_dim+5, self.word_vec_size),
        #                             nn.ReLU())
        self.fc = nn.Linear(self.num_dirs * self.hidden_size, self.vocab_size)

        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, sub_phrase_emb, loc_phrase_emb, enc_labels, dec_labels):
        """sub_phrase_emb, loc_phrase_emb, rel_phrase_emb.shape = (sent_num, 512), labels.shape = (sent_num, sent_length)"""
        slr_embeded = torch.cat([sub_phrase_emb, loc_phrase_emb], 1)
        slr_embeded = self.slr_mlp(slr_embeded)

        seq_len = enc_labels.size(1)  # (sent_num, 512)
        label_mask = (dec_labels != 0).float()
        batchsize = enc_labels.size(0)

        if self.variable_lengths:
            input_lengths = (enc_labels != 0).sum(1)
            input_lengths_list = input_lengths.data.cpu().numpy().tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist()
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist()
            s2r = {s: r for r, s in enumerate(sort_ixs)}
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]

            assert max(input_lengths_list) == enc_labels.size(1)

            sort_ixs = enc_labels.data.new(sort_ixs).long()
            recover_ixs = enc_labels.data.new(recover_ixs).long()

            input_labels = enc_labels[sort_ixs]

        slr_embeded = slr_embeded.view(batchsize, 1, -1)

        embedded = self.embedding(input_labels)  # (n, seq_len, word_embedding_size)
        embedded = self.input_dropout(embedded)  # (n, seq_len, word_embedding_size)
        embedded = self.mlp(embedded)  # (n, seq_len, word_embedding_size)

        slr_embedded = torch.cat([embedded, torch.cat([slr_embeded, torch.zeros(batchsize, seq_len - 1,
                                                                            self.word_embedding_size).cuda()], 1)], 2)

        if self.variable_lengths:
            slr_embedded = nn.utils.rnn.pack_padded_sequence(slr_embedded, sorted_input_lengths_list, batch_first=True)

        output, hidden = self.rnn(slr_embedded)

        # recover
        if self.variable_lengths:
            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # (batch, max_len, hidden)
            output = output[recover_ixs]

        output = output.view(batchsize * seq_len, -1)  # (batch*max_len, hidden)
        output = self.fc(output)  # (batch*max_len, vocab_size)

        dec_labels = dec_labels.view(-1)
        label_mask = label_mask.view(-1)

        lang_rec_loss = self.cross_entropy(output, dec_labels)
        lang_rec_loss = torch.sum(lang_rec_loss * label_mask) / torch.sum(label_mask)

        return lang_rec_loss


class VisualLangReconstructLoss(nn.Module):
    def __init__(self, opt):
        super(VisualLangReconstructLoss, self).__init__()
        self.use_weight = opt['use_weight']

    def forward(self, sub_phrase_emb, sub_phrase_recons, loc_phrase_emb, loc_phrase_recons, weights):
        """
        (sub_phrase_emb, sub_phrase_recons, loc_phrase_emb, loc_phrase_recons, rel_phrase_emb, rel_phrase_recons).shape=(sent_num, 512)
        weights.shape = (sent_num, 3)
        """
        # pdb.set_trace()
        sub_loss = self.mse_loss(sub_phrase_recons, sub_phrase_emb).sum(1).unsqueeze(1)  # (sent_num, 1)
        loc_loss = self.mse_loss(loc_phrase_recons, loc_phrase_emb).sum(1).unsqueeze(1)  # (sent_num, 1)
        if self.use_weight > 0:
            total_loss = (weights * torch.cat([sub_loss, loc_loss], 1)).sum(1).mean(0)
        else:
            total_loss = torch.cat([sub_loss, loc_loss], 1).sum(1).mean(0)
        return total_loss

    def mse_loss(self, recons, emb):
        return (recons-emb)**2

class LocationEncoder(nn.Module):
    def __init__(self, opt):
        super(LocationEncoder, self).__init__()
        init_norm = opt.get('visual_init_norm', 20)
        self.lfeats_normalizer = Normalize_Scale(5, init_norm)
        # self.dif_lfeat_normalizer = Normalize_Scale(25, init_norm)
        # self.fc = nn.Linear(5 + 25, opt['jemb_dim'])

    def forward(self, lfeats):
        sent_num, ann_num = lfeats.size(0), lfeats.size(1)
        concat = self.lfeats_normalizer(lfeats.contiguous().view(-1, 5))
        output = concat.view(sent_num, ann_num, 5)  # (sent_num, ann_num, 5)
        # output = self.fc(output)
        return output

class LocationDecoder(nn.Module):
    def __init__(self, opt):
        super(LocationDecoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(5, opt['jemb_dim']))
        # self.mlp = nn.Sequential(nn.Linear(opt['jemb_dim'], opt['jemb_dim']))

    def forward(self, loc_feats, total_ann_score):
        total_ann_score = total_ann_score.unsqueeze(1)  # (sent_num, 1, ann_num)
        loc_feats_fuse = torch.bmm(total_ann_score, loc_feats)  # (sent_num, 1, 512)
        loc_feats_fuse = loc_feats_fuse.squeeze(1)  # (sent_num, 512)
        loc_feats_fuse = self.mlp(loc_feats_fuse)  # (sent_num, 512)
        return loc_feats_fuse

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

        self.weight_fc = nn.Linear(self.num_layers * self.num_dirs *self.hidden_size, 2)

        self.sub_attn = PhraseAttention(self.hidden_size * self.num_dirs)
        self.loc_attn = PhraseAttention(self.hidden_size * self.num_dirs)

        self.rnn_decoder = RNNDncoder(opt)

        self.sub_encoder = SubjectEncoder(opt)
        self.loc_encoder = LocationEncoder(opt)


        self.sub_sim_attn = SimAttention(self.pool5_dim + self.fc7_dim, self.jemb_dim)


        self.sub_score = Score(self.pool5_dim+self.fc7_dim, opt['word_vec_size'],
                               opt['jemb_dim'])
        self.loc_score = Score(5, opt['word_vec_size'],
                               opt['jemb_dim'])

        self.att_res_weight = opt['att_res_weight']
        self.mse_loss = nn.MSELoss()

        self.sub_decoder = SubjectDecoder(opt)
        self.loc_decoder = LocationDecoder(opt)

        self.att_res_loss = AttributeReconstructLoss(opt)
        self.vis_res_loss = VisualLangReconstructLoss(opt)
        self.lang_res_loss = LangLangReconstructLoss(opt)
        self.rec_loss = ReconstructionLoss(opt)

        self.feat_fuse = nn.Sequential(
            nn.Linear(self.fc7_dim + self.pool5_dim + 5, opt['jemb_dim']))

    def forward(self, pool5_feats, fc7_feats, lfeats, labels, enc_labels, dec_labels,
                sub_sim, sub_emb, att_labels, select_ixs, att_weights):

        sent_num = pool5_feats.size(0)
        ann_num =  pool5_feats.size(1)
        label_mask = (dec_labels != 0).float()

        # 语言特征
        context, hidden, embedded = self.rnn_encoder(labels) # (sent_num, 10, 1024), (sent_num, 1024), (sent_num, 10, 512)

        weights = F.softmax(self.weight_fc(hidden)) # (sent_num, 3)

        sub_attn, sub_phrase_emb = self.sub_attn(context, embedded, labels) # (sent_num, 10), (sent_num, 512)
        loc_attn, loc_phrase_emb = self.loc_attn(context, embedded, labels) # (sent_num, 10), (sent_num, 512)

        # subject feats
        sub_feats = self.sub_encoder(pool5_feats, fc7_feats)  # (sent_num, ann_num, 2048+1024)
        # sub_feats = feats
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
        loc_feats = self.loc_encoder(lfeats)  # (sent_num, ann_num, 5+25)

        sub_ann_attn = self.sub_score(sub_feats, sub_phrase_emb)  # (sent_num, ann_num, 1)
        loc_ann_attn = self.loc_score(loc_feats, loc_phrase_emb)  # (sent_num, ann_num, 1)

        weights_expand = weights.unsqueeze(1).expand(sent_num, ann_num, 2)
        total_ann_score = (weights_expand * torch.cat([sub_ann_attn, loc_ann_attn], 2)).sum(2)  # (sent_num, ann_num)
        # if self.dist_pel > 0:
        #     total_ann_score = total_ann_score * dist

        sub_phrase_recons = self.sub_decoder(sub_feats, total_ann_score)  # (sent_num, 512)
        loc_phrase_recons = self.loc_decoder(loc_feats, total_ann_score)  # (sent_num, 512)

        loss = 0
        att_res_loss = 0
        lang_res_loss = 0
        vis_res_loss = 0
        # 第一次输出时，发现vis_res_loss=133.2017, att_res_loss = 4.3362, lan_rec_loss = 7.6249
        if self.vis_res_weight > 0:
            vis_res_loss = self.vis_res_loss(sub_phrase_emb, sub_phrase_recons, loc_phrase_emb,
                                             loc_phrase_recons, weights)
            loss = self.vis_res_weight * vis_res_loss

        if self.lang_res_weight > 0:
            lang_res_loss = self.lang_res_loss(sub_phrase_emb, loc_phrase_emb, enc_labels,
                                               dec_labels)

            loss += self.lang_res_weight * lang_res_loss

            # loss += torch.log(torch.div(lang_res_loss, vis_res_loss))
        # combined_loss
        loss = self.loss_divided * loss

        ann_score = total_ann_score.unsqueeze(1)
        fuse_feats = torch.cat([sub_feats, loc_feats], 2)  # (sent_num, ann_num, 2048+1024+512+512)

        fuse_feats = torch.bmm(ann_score, fuse_feats)
        fuse_feats = fuse_feats.squeeze(1)
        fuse_feats = self.feat_fuse(fuse_feats)
        rec_loss = self.rec_loss(fuse_feats, enc_labels, dec_labels)
        loss += self.loss_combined * rec_loss

        loss = loss + sub_loss 

        if self.att_res_weight > 0:
            # attribute_feats.shape = (12,12,512),total_ann_score.shape = (12,12), att_labels.shape = (12,50), att_weights.shape = (50)
            att_scores, att_res_loss = self.att_res_loss(sub_feats, total_ann_score, att_labels, select_ixs, att_weights)
            loss += self.att_res_weight * att_res_loss

        return total_ann_score, loss, sub_idx
