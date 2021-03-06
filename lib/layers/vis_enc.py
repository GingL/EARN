from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import pdb


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


class LocationEncoder(nn.Module):
    def __init__(self, opt):
        super(LocationEncoder, self).__init__()
        init_norm = opt.get('visual_init_norm', 20)
        self.lfeats_normalizer = Normalize_Scale(5, init_norm)
        self.dif_lfeat_normalizer = Normalize_Scale(25, init_norm)
        self.fc = nn.Linear(5 + 25, opt['jemb_dim'])

    def forward(self, lfeats, dif_lfeats):
        sent_num, ann_num = lfeats.size(0), lfeats.size(1)
        concat = torch.cat([self.lfeats_normalizer(lfeats.contiguous().view(-1, 5)),
                            self.dif_lfeat_normalizer(dif_lfeats.contiguous().view(-1, 25))], 1)
        output = concat.view(sent_num, ann_num, 5 + 25)  # (sent_num, ann_num, 5+25)
        # output = self.fc(output)
        return output


class SubjectEncoder(nn.Module):
    def __init__(self, opt):
        super(SubjectEncoder, self).__init__()
        self.word_vec_size = opt['word_vec_size']
        self.jemb_dim = opt['jemb_dim']
        self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']

        self.pool5_normalizer = Normalize_Scale(opt['pool5_dim'], opt['visual_init_norm'])
        self.fc7_normalizer = Normalize_Scale(opt['fc7_dim'], opt['visual_init_norm'])
        self.att_normalizer = Normalize_Scale(opt['jemb_dim'], opt['visual_init_norm'])
        self.phrase_normalizer = Normalize_Scale(opt['word_vec_size'], opt['visual_init_norm'])
        self.att_fuse = nn.Sequential(nn.Linear(opt['pool5_dim'] + opt['fc7_dim'], opt['jemb_dim']),
                                      nn.BatchNorm1d(opt['jemb_dim']))

    def forward(self, pool5, fc7):
        # fc7.shape = (sent_num,ann_num,2048,7,7), pool5.shape=(sent_num,ann_num,1024,7,7)
        sent_num, ann_num, grids = pool5.size(0), pool5.size(1), pool5.size(3) * pool5.size(4)
        batch = sent_num * ann_num

        pool5 = pool5.contiguous().view(batch, self.pool5_dim, -1)  # (sent_num * ann_num, 1024, 49)
        pool5 = pool5.transpose(1, 2).contiguous().view(-1, self.pool5_dim)  # (sent_num * ann_num * 49, 1024)
        pool5 = self.pool5_normalizer(pool5)  # (sent_num * ann_num * 49, 1024)
        pool5 = pool5.view(sent_num, ann_num, 49, -1).transpose(2, 3).contiguous().mean(3)  # (sent_num, ann_num, 1024)

        fc7 = fc7.contiguous().view(batch, self.fc7_dim, -1)  # (sent_num * ann_num, 2048, 49)
        fc7 = fc7.transpose(1, 2).contiguous().view(-1, self.fc7_dim)  # (sent_num * ann_num * 49, 2048)
        fc7 = self.fc7_normalizer(fc7)  # (sent_num * ann_num * 49, 2048)
        fc7 = fc7.view(sent_num, ann_num, 49, -1).transpose(2, 3).contiguous().mean(3)  # (sent_num, ann_num, 2048)

        avg_att_feats = torch.cat([pool5, fc7], 2)  # (sent_num, ann_num, 2048+1024)

        return avg_att_feats

class RelationEncoder(nn.Module):
    def __init__(self, opt):
        super(RelationEncoder, self).__init__()
        self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']
        self.fc7_normalizer = Normalize_Scale(opt['fc7_dim'], opt['visual_init_norm'])
        self.lfeat_normalizer    = Normalize_Scale(5, opt['visual_init_norm'])
        self.fc = nn.Linear(opt['fc7_dim']+5, opt['jemb_dim'])

    def forward(self, cxt_feats, cxt_lfeats, obj_attn, wo_obj_idx, dist):
        max_sim, max_id = torch.max(obj_attn, 1)
        sent_num = max_id.size(0)
        ann_num = cxt_feats.size(0)
        batch = sent_num*ann_num
        filtered_idx = max_sim.eq(0) #(sent_num)

        rel_feats = []
        rel_lfeats = []
        dists = []
        for i in range(max_id.size(0)):
            max_cxt_feats = cxt_feats[max_id[i]] #(2048)
            max_cxt_lfeats = cxt_lfeats[:,max_id[i],:] #(ann_num, 5)
            distance = dist[:, max_id[i], :] #(ann_num, 1)
            rel_feats.append(max_cxt_feats)
            rel_lfeats.append(max_cxt_lfeats)
            dists.append(distance)
        rel_feats = torch.stack(rel_feats) #(sent_num, 2048)
        rel_lfeats = torch.stack(rel_lfeats) #(sent_num, ann_num, 5)
        dists = torch.stack(dists) #(sent_num, ann_num, 1)
        rel_feats = rel_feats.unsqueeze(1).expand(sent_num, ann_num, self.fc7_dim) #(sent_num, ann_num, 2048)
        rel_feats[filtered_idx] = 0
        rel_lfeats[filtered_idx] = 0 #(sent_num, ann_num, 5)
        dists[filtered_idx] = 100 #(sent_num, ann_num, 1)

        rel_feats   = self.fc7_normalizer(rel_feats.contiguous().view(batch, -1)) # (sent_num * ann_num , 2048)
        rel_lfeats = self.lfeat_normalizer(rel_lfeats.contiguous().view(-1, 5))
        rel_feats_fuse = torch.cat([rel_feats, rel_lfeats], 1)
        # rel_feats_fuse = self.fc(concat)
        rel_feats_fuse = rel_feats_fuse.view(sent_num, ann_num, -1)
        return rel_feats_fuse, dists.squeeze(2), max_id