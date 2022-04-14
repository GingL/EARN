"""
args: imdb_name, net, iters, tag
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import json
import time
import numpy as np
import pprint
from scipy.misc import imread, imresize
import cv2

import torch
from torch.autograd import Variable

# mrcn imports
import _init_paths
from datasets.factory import get_imdb
from model.config import cfg, cfg_from_file, cfg_from_list
from model.bbox_transform import clip_boxes, bbox_transform_inv
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from utils.blob import im_list_to_blob
from utils.mask_utils import recover_masks
from pycocotools import mask as COCOmask


import pdb
# mrcn dir
mrcn_dir = osp.join('/home/xuejing_liu/yuki/MattNet', 'pyutils', 'mask-faster-rcnn')


def get_imdb_name(imdb_name):
    if imdb_name in ['refcoco', 'refcocog']:
        return {'TRAIN_IMDB': '%s_train+%s_val' % (imdb_name, imdb_name),
                'TEST_IMDB': '%s_test' % imdb_name}
    elif imdb_name == 'coco_minus_refer':
        return {'TRAIN_IMDB': "coco_2014_train_minus_refer_valtest+coco_2014_valminusminival",
                'TEST_IMDB': "coco_2014_minival"}


class Inference:
    def __init__(self, args):

        self.imdb_name = args.imdb_name
        self.net_name = args.net_name
        self.tag = args.tag
        self.iters = args.iters

        # Config
        cfg_file = osp.join(mrcn_dir, 'experiments/cfgs/%s.yml' % self.net_name)
        cfg_list = ['ANCHOR_SCALES', [4, 8, 16, 32], 'ANCHOR_RATIOS', [0.5, 1, 2]]
        if cfg_file is not None: cfg_from_file(cfg_file)
        if cfg_list is not None: cfg_from_list(cfg_list)
        print('Using config:')
        pprint.pprint(cfg)

        # load imdb
        self.imdb = get_imdb(get_imdb_name(self.imdb_name)['TEST_IMDB'])

        # Load network
        self.net = self.load_net()

    def load_net(self):
        # Load network
        if self.net_name == 'vgg16':
            net = vgg16(batch_size=1)
        elif self.net_name == 'res101':
            net = resnetv1(batch_size=1, num_layers=101)
        else:
            raise NotImplementedError

        # 暂时未找到create_architecture
        net.create_architecture(self.imdb.num_classes, tag='default',
                                anchor_scales=cfg.ANCHOR_SCALES,
                                anchor_ratios=cfg.ANCHOR_RATIOS)
        net.eval()
        net.cuda()

        # Load model
        model = osp.join(mrcn_dir, 'output/%s/%s/%s/%s_mask_rcnn_iter_%s.pth' % \
                         (self.net_name, get_imdb_name(self.imdb_name)['TRAIN_IMDB'], self.tag, self.net_name,
                          self.iters))
        assert osp.isfile(model)
        net.load_state_dict(torch.load(model))
        print('pretrained-model loaded from [%s].' % model)
        net.train()
        lr = cfg.TRAIN.LEARNING_RATE
        params = []
        for key, value in dict(net.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1)*0.1, 'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params':[value],'lr':lr*0.1, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        self.optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
        return net

    def extract_head(self, img_path, train_mode):
        # extract head (1, 1024, im_height*scale/16.0, im_width*scale/16.0) in Variable cuda float
        # and im_info [[ih, iw, scale]] in float32 ndarray
        if train_mode == 'test':
            self.net.eval()
        im = cv2.imread(img_path)
        blobs, im_scales = self._get_blobs(im)
        # _相当于net._layers["head"](Variable(torch.from_numpy(image.transpose([0,3,1,2])).cuda(), volatile=True))
        # head_feat.shape = (1, 1024, im_height*scale/16.0, im_width*scale/16.0)
        head_feat = self.net.extract_head(blobs['data'])
        im_info = np.array([[blobs['data'].shape[1], blobs['data'].shape[2], im_scales[0]]])
        return head_feat, im_info.astype(np.float32)

    def box_to_spatial_fc7(self, net_conv, im_info, ori_boxes, train_mode):
        """
    Arguments:
      net_conv (Variable)  : (1, 1024, H, W)
      im_info (float32)    : [[ih, iw, scale]]
      ori_boxes (float32)  : (n, 4) [x1y1x2y2]
    Returns:
      pool5 (float)        : (n, 1024, 7, 7)
      spatial_fc7 (float)  : (n, 2048, 7, 7)
    """
        self.net.train()
        self.net._mode = 'Train'

        if train_mode == 'test':
            self.net.eval()
            self.net._mode = 'TEST'



        # make rois
        batch_inds = Variable(net_conv.data.new(ori_boxes.shape[0], 1).zero_())
        scaled_boxes = (ori_boxes * im_info[0][2]).astype(np.float32)
        scaled_boxes = Variable(torch.from_numpy(scaled_boxes).cuda())
        rois = torch.cat([batch_inds, scaled_boxes], 1)

        # pool fc7
        if cfg.POOLING_MODE == 'crop':
            pool5 = self.net._crop_pool_layer(net_conv, rois)
        else:
            pool5 = self.net._roi_pool_layer(net_conv, rois)  # (n, 1024, 7, 7)

        spatial_fc7 = self.net.resnet.layer4(pool5)  # (n, 2048, 7, 7), equavalent to _head_to_tail
        return pool5, spatial_fc7


    def _get_image_blob(self, im):
        """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
        # pdb.set_trace()
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        # if im_shape=(320, 480, 3) then im_shape[0:2]=(320,480)
        # then im_size_min =320, im_size_max = 480
        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def _get_blobs(self, im):
        """Convert an image and RoIs within that image into network inputs."""
        blobs = {}
        blobs['data'], im_scale_factors = self._get_image_blob(im)

        return blobs, im_scale_factors

    def box_to_pool5_fc7(self, net_conv, im_info, ori_boxes, train_mode):
        """
    Arguments:
      net_conv (Variable)  : (1, 1024, H, W)
      im_info (float32)    : [[ih, iw, scale]]
      ori_boxes (float32)  : (n, 4) [x1y1x2y2]
    Returns:
      pool5 (float): (n, 1024)
      fc7 (float)  : (n, 2048)
    """
        self.net.train()
        self.net._mode = 'TRAIN'
        if train_mode == 'test':
            self.net.eval()
            self.net._mode = 'TEST'        

        # make rois
        batch_inds = Variable(net_conv.data.new(ori_boxes.shape[0], 1).zero_())
        scaled_boxes = (ori_boxes * im_info[0][2]).astype(np.float32)
        scaled_boxes = Variable(torch.from_numpy(scaled_boxes).cuda())
        rois = torch.cat([batch_inds, scaled_boxes], 1)

        # pool fc7
        if cfg.POOLING_MODE == 'crop':
            pool5 = self.net._crop_pool_layer(net_conv, rois)
        else:
            pool5 = self.net._roi_pool_layer(net_conv, rois) # (n,1024,7,7)

        fc7 = self.net._head_to_tail(pool5)  # (n, 2048, 7, 7)
        pool5 = pool5.mean(3).mean(2) # (n, 1024)
        fc7 = fc7.mean(3).mean(2)  # (n, 2048)
        return pool5, fc7
