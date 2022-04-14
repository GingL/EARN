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

from tqdm import tqdm
from multiprocessing import Pool

def yxyx_to_xyxy(boxes):
    """Convert [y1 x1 y2 x2] box format to [x1 y1 x2 y2] format."""
    return boxes[:,[1,0,3,2]]

# IoU function
def computeIoU(box1, box2):
    # each box is of [x1, y1, x2, y2]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (box1[2]-box1[0]+1) * (box1[3]-box1[1]+1) + (box2[2]-box2[0]+1) * (box2[3]-box2[1]+1) - inter
    return float(inter) / union

def eval_split(dataset, loader, model, opt):
    torch.set_grad_enabled(False)
    model.eval()

    loss_sum = 0
    loss_evals = 0
    acc = 0
    predictions = []

    for batch in loader:
        # for key, value in batch.items():
        #     print(key, len(value))
        tic = time.time()
        att_weights = dataset.get_attribute_weights()
        ref_ids = batch['ref_id']
        pool5_feats, fc7_feats, lfeats = batch['pool5_feats'].cuda(), batch['fc7_feats'].cuda(), batch['lfeats'].cuda()
        labels = batch['labels'].cuda()
        max_len = (labels != 0).sum(1).max().data[0]
        labels = labels[:, :max_len] 

        start_words = np.ones([labels.size(0), 1], dtype=int) * (dataset.word_to_ix['<s>'])
        start_words = Variable(torch.from_numpy(start_words).long().cuda())
        enc_labels = labels.clone()
        enc_labels = torch.cat([start_words, enc_labels], 1)

        zero_pad = np.zeros([labels.size(0), 1], dtype=int)
        zero_pad = Variable(torch.from_numpy(zero_pad).long().cuda())
        dec_labels = labels.clone()
        dec_labels = torch.cat([dec_labels, zero_pad], 1)    

        image_id = batch['image_id']
        sub_sim = batch['sub_sim'].cuda()
        sub_emb = batch['sub_emb'].cuda()
        att_labels, select_ixs = batch['att_labels'].cuda(), batch['select_ixs'].cuda()

        toc = time.time()
        # print(toc-tic)

        tic = time.time()
        scores, loss, sub_idxs = model(pool5_feats, fc7_feats, lfeats, labels, enc_labels, dec_labels, sub_sim, sub_emb, att_labels,
                                                  select_ixs, att_weights)
        toc = time.time()
        # print(toc - tic)
        scores = scores.squeeze(0)
        loss = loss.data[0].item()
        loss_sum += loss

        gd_boxes = batch['gd_boxes']

        for i, ref_id in enumerate(ref_ids):
            score = scores[i]
            sub_idx = sub_idxs[i]
            box_num = len(dataset.Refs[ref_id]['bbxes'])
            if box_num < 101:
                score = score[0:box_num]
            pred_ix = torch.argmax(score)
            k = 2
            while sub_idx[pred_ix] == 0:
                maxk, idx = torch.topk(score, k)
                pred_ix = idx[k - 1].data.cpu()
                k += 1
                if k > score.size(0):
                    break

            # pred_box = dataset.Refs[ref_id]['bbxes'][pred_ix]
            ann_boxes = yxyx_to_xyxy(np.vstack([box for box in dataset.Refs[ref_id]['bbxes']])[0:100, :])
            gd_box = dataset.Refs[ref_id]['bbxes'][-1]
            ann_boxes = np.vstack([ann_boxes, gd_box])
            pred_box = ann_boxes[pred_ix]

            gd_box = np.float64(gd_boxes[i])
            IoU = computeIoU(pred_box, gd_box)
            if IoU >= 0.5:
                acc += 1
            loss_evals += 1
            entry = {}
            entry['image_id'] = image_id[i]
            entry['ref_id'] = ref_id
            entry['sent'] = dataset.decode_labels(labels[i].data.cpu().numpy())  # gd-truth sent
            entry['gd_box'] = gd_box.tolist()
            entry['pred_box'] = pred_box.tolist()
            entry['IoU'] = IoU.tolist()
            print('evaluating [val] ... ref[%s]\'s sents, acc=%.2f%%, (%.4f)' % \
                  (ref_id, acc*100.0/loss_evals, loss))


            predictions.append(entry)

    torch.set_grad_enabled(True)
    return loss_sum / loss_evals, acc / loss_evals, predictions


# def eval_split(loader, model, split, opt):

#     verbose = opt.get('verbose', True)
#     num_sents = opt.get('num_sents', -1)
#     assert split != 'train', 'Check the evaluation split.'

#     model.eval()

#     loader.resetIterator(split)
#     loss_sum = 0
#     loss_evals = 0
#     acc = 0
#     predictions = []
#     finish_flag = False
#     model_time = 0

#     while True:
#         data = loader.getTestBatch(split, opt)
#         att_weights = loader.get_attribute_weights()
#         ref_ids = data['ref_ids']
#         Feats = data['Feats']
#         labels = data['labels']
#         enc_labels = data['enc_labels']
#         dec_labels = data['dec_labels']
#         image_id = data['image_id']

#         att_labels, select_ixs = data['att_labels'], data['select_ixs']
#         sim = data['sim']

#         tic = time.time()
#         scores, loss, sub_idx = model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], labels, enc_labels, dec_labels, sim['sub_sim'],
#                                                   sim['sub_emb'],  att_labels,
#                                                   select_ixs, att_weights)

#         scores = scores.squeeze(0)
#         sub_idx = sub_idx.squeeze(0)
#         loss = loss.data[0].item()

#         pred_ix = torch.argmax(scores)

#         k = 2
#         while sub_idx[pred_ix] == 0:
#             maxk, idx = torch.topk(scores, k)
#             pred_ix = idx[k - 1].data.cpu()
#             k += 1
#             if k > scores.size(0):
#                 break

#         pred_box = loader.Refs[ref_ids]['bbxes'][pred_ix]

#         gd_box = data['gd_boxes']
#         loss_sum += loss
#         loss_evals += 1

#         IoU = computeIoU(pred_box, gd_box)

#         if IoU >= 0.5:
#             acc += 1

#         entry = {}
#         entry['image_id'] = image_id
#         entry['ref_id'] = ref_ids
#         entry['sent'] = loader.decode_labels(labels.data.cpu().numpy())[0]  # gd-truth sent
#         entry['gd_box'] = gd_box
#         entry['pred_score'] = scores.tolist()[pred_ix]
#         entry['IoU'] = IoU

#         predictions.append(entry)
#         toc = time.time()
#         model_time += (toc - tic)

#         if num_sents > 0  and loss_evals >= num_sents:
#             finish_flag = True
#             break
#         ix0 = data['bounds']['it_pos_now']
#         ix1 = data['bounds']['it_max']
#         # if ix0== 974:
#         #    print(1)
#         if verbose:
#             print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%, (%.4f)' % \
#                   (split, ix0, ix1, acc*100.0/loss_evals, loss))
#         model_time = 0

#         if finish_flag or data['bounds']['wrapped']:
#             break

#     return loss_sum / loss_evals, acc / loss_evals, predictions


