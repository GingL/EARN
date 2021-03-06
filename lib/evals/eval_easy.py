from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import json
import h5py
import time
from pprint import pprint

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pdb


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
            dist = Feats['dist']
            cxt_fc7 = Feats['cxt_fc7']
            cxt_lfeats = Feats['cxt_lfeats']
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
                                 cxt_lfeats, label, enc_label, dec_label, sub_sim, obj_sim,
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


