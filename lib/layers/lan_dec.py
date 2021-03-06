from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb

class RNNDncoder(nn.Module):
    def __init__(self, opt):
        # word_embedding_size = 512, word_vec_size = 512, rnn_hidden_size = 512, bidirectional = True
        # word_drop_out = 0.5, rnn_drop_out = 0.2, rnn_num_layers = 1, rnn_type = lstm, variable_lengths = True
        super(RNNDncoder, self).__init__()

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
        self.rnn = getattr(nn, self.rnn_type.upper())(self.word_vec_size*2, self.hidden_size, self.n_layers,
                                                      batch_first=True, bidirectional=self.bidirectional,
                                                      dropout=self.dropout_p)
        self.num_dirs = 2 if self.bidirectional else 1

        self.slr_mlp = nn.Sequential(nn.Linear(self.word_vec_size * 3, self.word_vec_size),
                                     nn.ReLU())
        self.fc = nn.Linear(self.num_dirs * self.hidden_size, self.vocab_size)

    def forward(self, vis_att_fuse, enc_labels):
        # vis_att_fuse (sent_num, 512) enc_labels (sent_num, sent_length)
        seq_len = enc_labels.size(1)
        sent_num = enc_labels.size(0)

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

        # vis_att_fuse = vis_att_fuse.unsqueeze(0) # (1, sent_num, 512)
        vis_att_fuse = vis_att_fuse.unsqueeze(1)  # (sent_num, 1, 512)
        embedded = self.embedding(input_labels)  # (n, seq_len, word_embedding_size)
        embedded = self.input_dropout(embedded)  # (n, seq_len, word_embedding_size)
        embedded = self.mlp(embedded)  # (n, seq_len, word_vec_size)

        embedded = torch.cat([embedded, torch.cat([vis_att_fuse, torch.zeros(sent_num, seq_len - 1,
                                                                            self.word_vec_size).cuda()], 1)], 2)
        # embedded = torch.cat([torch.cat([vis_att_fuse, torch.zeros(sent_num, seq_len - 1,
        #                                                                     self.word_vec_size).cuda()], 1)], 2)


        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)

        output, hidden = self.rnn(embedded)
        # output, hidden = self.rnn(embedded, (vis_att_fuse, torch.zeros_like(vis_att_fuse)))

        # recover
        if self.variable_lengths:
            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # (batch, max_len, hidden)
            output = output[recover_ixs]

        output = output.view(sent_num * seq_len, -1)  # (batch*max_len, hidden)
        output = self.fc(output)  # (batch*max_len, vocab_size)

        return output

class LocationDecoder(nn.Module):
    def __init__(self, opt):
        super(LocationDecoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(5 + 25, opt['jemb_dim']))
        # self.mlp = nn.Sequential(nn.Linear(opt['jemb_dim'], opt['jemb_dim']))

    def forward(self, loc_feats, total_ann_score):
        total_ann_score = total_ann_score.unsqueeze(1)  # (sent_num, 1, ann_num)
        loc_feats_fuse = torch.bmm(total_ann_score, loc_feats)  # (sent_num, 1, 512)
        loc_feats_fuse = loc_feats_fuse.squeeze(1)  # (sent_num, 512)
        loc_feats_fuse = self.mlp(loc_feats_fuse)  # (sent_num, 512)
        return loc_feats_fuse

class SubjectDecoder(nn.Module):
    def __init__(self, opt):
        super(SubjectDecoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(opt['pool5_dim'] + opt['fc7_dim'], opt['jemb_dim']))
        # self.mlp = nn.Sequential(nn.Linear(opt['jemb_dim'], opt['jemb_dim']))

    def forward(self, sub_feats, total_ann_score):
        total_ann_score = total_ann_score.unsqueeze(1)  # (sent_num, 1, ann_num)
        sub_feats_fuse = torch.bmm(total_ann_score, sub_feats)  # (sent_num, 1, 2048+512)
        sub_feats_fuse = sub_feats_fuse.squeeze(1)
        sub_feats_fuse = self.mlp(sub_feats_fuse)
        return sub_feats_fuse

class RelationDecoder(nn.Module):
    def __init__(self, opt):
        super(RelationDecoder, self).__init__()
        self.jemb_dim = opt['jemb_dim']
        self.word_vec_size = opt['word_vec_size']
        self.fc7_dim = opt['fc7_dim']
        # self.mlp = nn.Sequential(nn.Linear(self.jemb_dim, self.jemb_dim))
        self.mlp = nn.Sequential(nn.Linear(self.fc7_dim + 5, self.jemb_dim))

    def forward(self, rel_feats, total_ann_score):
        total_ann_score = total_ann_score.unsqueeze(1)  # (sent_num, 1, ann_num)
        rel_feats_fuse = torch.bmm(total_ann_score, rel_feats)
        rel_feats_fuse = rel_feats_fuse.squeeze(1)
        rel_feats_fuse = self.mlp(rel_feats_fuse)
        return rel_feats_fuse


