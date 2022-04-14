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
import tqdm

# model
import _init_paths
from mrcn import inference_no_imdb
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from multiprocessing import Pool
import pdb


# box functions
def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

def yxyx_to_xyxy(boxes):
    """Convert [y1 x1 y2 x2] box format to [x1 y1 x2 y2] format."""
    return boxes[:,[1,0,3,2]]


class Loader(object):
    def __init__(self, data_json, similarity, data_h5=None, data_emb=None):
        print('Loader loading data.json: ', data_json)
        self.info = json.load(open(data_json))
        self.word_to_ix = self.info['word_to_ix']
        self.ix_to_word = {ix: wd for wd, ix in self.word_to_ix.items()}
        print('vocab size is ', self.vocab_size)
        self.refs = self.info['refs']
        print('we have %s refs.' % len(self.refs))

        # construct mapping
        self.Refs = {ref['ann_id']: ref for ref in self.refs}

    # @property装饰器负责把一个方法变成属性调用,广泛应用在类的定义中，可以让调用者写出简短的代码，同时保证对参数进行必要的检查
    @property
    def vocab_size(self):
        # len(self.word_to_ix) == 1999
        return len(self.word_to_ix)

    @property
    def label_length(self):
        return self.info['label_length']

class DataLoader(Loader):

    def __init__(self, data_json, data_h5, similarity, opt):
        # parent loader instance
        Loader.__init__(self, data_json, similarity, data_h5)

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

    def fetch_grid_feats(self, boxes, net_conv, im_info):
        """returns -pool5 (n, 1024, 7, 7) -fc7 (n, 2048, 7, 7)"""
        pool5, fc7 = self.mrcn.box_to_spatial_fc7(net_conv, im_info, boxes)
        return pool5, fc7

    def getFeature(self, opt):
        pool5_dim, fc7_dim = opt['pool5_dim'], opt['fc7_dim']

        for ref in self.refs:
            ref_id = ref['ann_id']
            image_id = ref['image_id']            
            
            # fetch image features
            head, im_info = self.image_to_head(image_id)
            head = Variable(torch.from_numpy(head).cuda())

            # fetch ann features
            # pdb.set_trace()
            ann_boxes = yxyx_to_xyxy(np.vstack([box for box in self.Refs[ref_id]['bbxes']])[0:100, :])
            gd_box = self.Refs[ref_id]['bbxes'][-1]
            ann_boxes = np.vstack([ann_boxes, gd_box])
            ann_pool5, ann_fc7 = self.fetch_grid_feats(ann_boxes, head, im_info)

            if ann_pool5.size(0) < 101:
                zero_pad = torch.zeros((101-ann_pool5.size(0)), 1024, 7, 7).cuda()
                ann_pool5 = torch.cat([ann_pool5, zero_pad], 0)

            if ann_fc7.size(0) < 101:
                zero_pad = torch.zeros((101-ann_fc7.size(0)), 2048, 7, 7).cuda()
                ann_fc7 = torch.cat([ann_fc7, zero_pad], 0)

            pool5 = ann_pool5.contiguous().view(101, pool5_dim, -1)
            pool5 = nn.functional.normalize(pool5, p=2, dim=1)
            pool5,_ = pool5.max(2)

            fc7 = ann_fc7.contiguous().view(101, fc7_dim, -1)
            fc7 = nn.functional.normalize(fc7, p=2, dim=1)
            fc7,_ = fc7.max(2)
            
            file_path = osp.join('/home/xuejing_liu/yuki/cache/feats', 'flickr30k', 'fc7_feats')
            if not osp.isdir(file_path):
                os.makedirs(file_path)
            file_name = osp.join(file_path, str(ref_id)+'.pt')
#            if os.path.exists(file_name):
#                raise Exception('File already exists!')
            torch.save(fc7, file_name)

            file_path = osp.join('/home/xuejing_liu/yuki/cache/feats', 'flickr30k', 'pool5_feats')
            if not osp.isdir(file_path):
                os.makedirs(file_path)
            file_name = osp.join(file_path, str(ref_id)+'.pt')
            torch.save(pool5, file_name)

            print("%s genetated" %(str(ref_id)))

def main(args):
    RootPath = '/home/xuejing_liu/yuki/MattNet'
    opt = vars(args)
    # initialize
    opt['dataset_splitBy'] = opt['dataset']

    # set random seed
    # seed()  用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed()
    # 值，则每次生成的随机数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
    torch.manual_seed(opt['seed'])
    random.seed(opt['seed'])

    # set up loader
    data_json = osp.join(RootPath, 'cache/prepro', opt['dataset_splitBy'], 'data2.json')
    loader = DataLoader(data_h5=None, data_json=data_json, similarity=None, opt=opt)

    # prepare feats
    feats_dir = '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
    head_feats_dir = osp.join(RootPath, 'cache/feats/', opt['dataset_splitBy'], 'mrcn', feats_dir)

    loader.prepare_mrcn(head_feats_dir, args)


    # set up model
    opt['vocab_size'] = loader.vocab_size
    opt['fc7_dim'] = loader.fc7_dim
    opt['pool5_dim'] = loader.pool5_dim


    data = loader.getFeature(opt)
    print('done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='flickr30k', help='name of dataset')
    parser.add_argument('--checkpoint_path', type=str, default='output_cls', help='directory to save models')
    parser.add_argument('--exp_id', type=str, default='exp2', help='experiment id')
    parser.add_argument('--gpuid', type=int, default=0, help='which gpu to use, -1 = use CPU id')
    parser.add_argument('--learning_rate_decay_start', type=int, default=4000, help='at what iter to start decaying learning rate')
    parser.add_argument('--learning_rate_decay_every', type=int, default=4000, help='every how many iters thereafter to drop LR by half')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8, help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--learning_rate', type=float, default=4e-5, help='learning rate')
    parser.add_argument('--optim_alpha', type=float, default=0.8, help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999, help='beta used for adam')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for adam')
    parser.add_argument('--losses_log_every', type=int, default=25, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--save_checkpoint_every', type=int, default=2000, help='how often to save a model checkpoint?')
    parser.add_argument('--max_iters', type=int, default=20000, help='max number of iterations to run') 
    parser.add_argument('--load_best_score', type=int, default=1, help='Do we load previous best score when resuming training.') 
    parser.add_argument('--grad_clip', type=float, default=0.1, help='clip gradients at this value')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size in number of images per batch')
    # Visual Encoder Setting
    parser.add_argument('--visual_sample_ratio', type=float, default=0.3, help='ratio of same-type objects over different-type objects')
    parser.add_argument('--visual_fuse_mode', type=str, default='concat', help='concat or mul')
    parser.add_argument('--visual_init_norm', type=float, default=20, help='norm of each visual representation')
    parser.add_argument('--visual_use_bn', type=int, default=-1, help='>0: use bn, -1: do not use bn in visual layer')    
    parser.add_argument('--visual_use_cxt', type=int, default=1, help='if we use contxt')
    parser.add_argument('--visual_cxt_type', type=str, default='frcn', help='frcn or res101')
    parser.add_argument('--visual_drop_out', type=float, default=0.2, help='dropout on visual encoder')
    parser.add_argument('--window_scale', type=float, default=2.5, help='visual context type')
    parser.add_argument('--with_visual_att', type=int, default=0, help='whether to use visual_att')
    # FRCN setting
    parser.add_argument('--imdb_name', default='coco_minus_refer', help='image databased trained on.')
    parser.add_argument('--net_name', default='res101', help='net_name: res101 or vgg16')
    parser.add_argument('--iters', default=1250000, type=int, help='iterations we trained for faster R-CNN')
    parser.add_argument('--tag', default='notime', help='on default tf, don\'t change this!')
   # Loss Setting
    parser.add_argument('--seed', type=int, default=24, help='random number generator seed to use')

    parser.add_argument('--split', type=str, default='val', help='split: testAB or val, etc')

    # argparse
    args = parser.parse_args()

    main(args)
