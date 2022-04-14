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
from loaders.data_loader_flickr30k import MyDataset
from layers.model_flickr30k import KARN
import evals.utils as model_utils
import evals.eval_easy_flickr30k as eval_utils
from opt import parse_opt
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter


def main(args):
    RootPath = '/home/xuejing_liu/yuki/ARN'
    opt = vars(args)
    # initialize
    opt['dataset_splitBy'] = opt['dataset']
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

    # # set up loader
    # data_json = osp.join(RootPath, 'cache/prepro', opt['dataset_splitBy'], 'data.json')
    # # data_h5 = osp.join(RootPath, 'cache/prepro', opt['dataset_splitBy'], 'data.h5')
    # similarity = osp.join(RootPath, 'cache/similarity', opt['dataset_splitBy'], 'similarity.json')
    # loader = DataLoader(data_h5=None, data_json=data_json, similarity=similarity, opt=opt)

    # set up loader
    data_json = osp.join(RootPath, 'cache/prepro', opt['dataset_splitBy'], 'data2.json')
    data_json = json.load(open(data_json))
    similarity = osp.join(RootPath, 'cache/similarity', opt['dataset_splitBy'], 'similarity2.pkl')
    data_sim = pickle.load(open(similarity, 'rb'))
    train_dataset = MyDataset(data_json, data_sim, 'train')
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4)

    val_dataset = MyDataset(data_json, data_sim, 'test')
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4)

    # # prepare feats
    # feats_dir = '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
    # head_feats_dir = osp.join(RootPath, 'cache/feats/', opt['dataset_splitBy'], 'mrcn', feats_dir)

    # loader.prepare_mrcn(head_feats_dir, args)


    # set up model
    opt['vocab_size'] = train_dataset.vocab_size
    opt['fc7_dim'] = 2048
    opt['pool5_dim'] = 1024
    opt['num_atts'] = train_dataset.num_atts

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
    # loader.iterators = infos.get('iterators', loader.iterators)
    if opt['load_best_score'] == 1:
        best_val_score = infos.get('best_val_score', None)

    att_weights = train_dataset.get_attribute_weights()

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
    while iter <= opt['max_iters']:
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            T = {}
            tic = time.time()
            labels = batch['labels'].cuda()
            max_len = (labels != 0).sum(1).max().data[0]
            labels = labels[:, :max_len]

            start_words = np.ones([labels.size(0), 1], dtype=int) * (train_dataset.word_to_ix['<s>'])
            start_words = Variable(torch.from_numpy(start_words).long().cuda())
            enc_labels = labels.clone()
            enc_labels = torch.cat([start_words, enc_labels], 1)

            zero_pad = np.zeros([labels.size(0), 1], dtype=int)
            zero_pad = Variable(torch.from_numpy(zero_pad).long().cuda())
            dec_labels = labels.clone()
            dec_labels = torch.cat([dec_labels, zero_pad], 1)

            pool5_feats, fc7_feats, lfeats = batch['pool5_feats'].cuda(), batch['fc7_feats'].cuda(), batch['lfeats'].cuda()
            sub_sim = batch['sub_sim'].cuda()
            sub_emb = batch['sub_emb'].cuda()
            att_labels, select_ixs = batch['att_labels'].cuda(), batch['select_ixs'].cuda()

            T['data'] = time.time() - tic

            tic = time.time()
            scores, loss, _ = model(pool5_feats, fc7_feats, lfeats, labels, enc_labels, dec_labels, sub_sim, sub_emb, att_labels, select_ixs, att_weights)

            loss.backward()
            model_utils.clip_gradient(optimizer, opt['grad_clip'])
            optimizer.step() 
            T['model'] = time.time() - tic

            data_time += T['data']
            model_time += T['model']

            if iter % opt['losses_log_every'] == 0:
                loss_history[iter] = (loss.data[0]).item()
                print('iter[%s], train_loss=%.3f, lr=%.2E, data:%.2fs/iter, model:%.2fs/iter' \
                    % (iter, loss.data[0].item(), lr, data_time / opt['losses_log_every'],
                        model_time / opt['losses_log_every']))
                writer.add_scalar('Train/Loss', loss.data[0].item(), iter)
                data_time, model_time = 0, 0

            if opt['learning_rate_decay_start'] > 0 and iter > opt['learning_rate_decay_start']:
                frac = (iter - opt['learning_rate_decay_start']) / opt['learning_rate_decay_every']
                decay_factor = 0.1 ** frac
                lr = opt['learning_rate'] * decay_factor
                model_utils.set_lr(optimizer, lr)

            if iter % opt['save_checkpoint_every'] == 0 or iter == opt['max_iters']:

                val_loss, acc, predictions = eval_utils.eval_split(val_dataset, val_loader, model, opt)
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
            if iter > opt['max_iters'] and opt['max_iters'] > 0:
                break

if __name__ == '__main__':
    args = parse_opt()
    main(args)

