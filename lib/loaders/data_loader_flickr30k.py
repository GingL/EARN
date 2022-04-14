"""
data_json has 
0. refs:       [{ref_id, ann_id, box, image_id, split, category_id, sent_ids, att_wds}]
1. images:     [{image_id, ref_ids, file_name, width, height, h5_id}]
2. anns:       [{ann_id, category_id, image_id, box, h5_id}]
3. sentences:  [{sent_id, tokens, h5_id}]
4. word_to_ix: {word: ix}
5. att_to_ix : {att_wd: ix}
6. att_to_cnt: {att_wd: cnt}
7. label_length: L

Note, box in [xywh] format
label_h5 has
/labels is (M, max_length) uint32 array of encoded labels, zeros padded
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re
import sys
import io

import os.path as osp
import numpy as np
import h5py
import json
import random

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import skimage.io

import _init_paths
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

def yxyx_to_xyxy(boxes):
    """Convert [y1 x1 y2 x2] box format to [x1 y1 x2 y2] format."""
    return boxes[:,[1,0,3,2]]

class MyDataset(Dataset):
    def __init__(self, data_json, similarity, split):
        self.info = data_json
        self.similarity = similarity
        self.word_to_ix = self.info['word_to_ix']
        self.ix_to_word = {ix: wd for wd, ix in self.word_to_ix.items()}

        print('vocab size is ', self.vocab_size)
        self.refs = self.info['refs']

        # 将数据集中自带的类别作为类别， 分类性能74%
        # i = 0
        # cls_wds = []
        # self.ix_to_cat = {}
        # self.cat_to_ix = {}
        # self.cat_to_cnt = {}
        # for ref in self.refs:
        #     cls_wd = ref['category_id']
        #     if cls_wd not in cls_wds:
        #         cls_wds.append(cls_wd)
        #         self.ix_to_cat[i]=cls_wd
        #         self.cat_to_ix[cls_wd]=i
        #         i += 1
        #     self.cat_to_cnt[cls_wd] = self.cat_to_cnt.get(cls_wd, 0) + 1

        # 将最后一个词作为类别，但只用前100个数目最多的
        self.cat_to_cnt = self.info['cat_to_cnt']
        self.ix_to_cat = self.info['ix_to_cat']
        self.cat_to_ix = self.info['cat_to_ix']


        print('object cateogry size is ', len(self.ix_to_cat))
        # print(self.ix_to_cat[0])
        print('label_length is ', self.label_length)

        # construct mapping
        self.Refs = {ref['ann_id']: ref for ref in self.refs}

        self.num_cats = len(self.cat_to_ix)
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

        self.split = split

        # img_iterators for each split
        self.split_ix = {}
        self.iterators = {}
        for ref_id, ref in self.Refs.items():
            # we use its ref's split (there is assumption that each image only has one split)
            ref_split = ref['split']
            if ref_split not in self.split_ix:
                self.split_ix[ref_split] = []
                self.iterators[ref_split] = 0
            self.split_ix[ref_split] += [ref_id]
        for k, v in self.split_ix.items():
            print('assigned %d expressions to split %s' %(len(v), k))

    @property
    def vocab_size(self):
        # len(self.word_to_ix) == 1999
        return len(self.word_to_ix)

    @property
    def label_length(self):
        return self.info['label_length']

    def __getitem__(self, idx):
        split_ix = self.split_ix[self.split]
        ref_id = split_ix[idx]
        ref = self.Refs[ref_id]
        image_id = ref['image_id']
        gd_boxes = ref['box']
        # load image features
        fc7_file_path = osp.join('/home/xuejing_liu/yuki/MattNet/cache/feats', 'flickr30k', 'fc7_feats')
        fc7_file_name = osp.join(fc7_file_path, str(ref_id)+'.pt')
        fc7_feats = torch.load(fc7_file_name, map_location="cpu")
        fc7_feats =fc7_feats.detach()
            
        pool5_file_path = osp.join('/home/xuejing_liu/yuki/MattNet/cache/feats', 'flickr30k', 'pool5_feats')
        pool5_file_name = osp.join(pool5_file_path, str(ref_id)+'.pt')
        pool5_feats = torch.load(pool5_file_name, map_location="cpu")
        pool5_feats = pool5_feats.detach()

        sub_sim = torch.Tensor(self.similarity[str(ref_id)]['sub_sim']).squeeze()
        sub_emb = torch.Tensor(self.similarity[str(ref_id)]['sub_emb']).squeeze()

        # if sub_sim.size(0) < 101:
        #     zero_pad = torch.zeros(100-sub_sim.size(0))
        #     sub_sim = torch.cat([sub_sim, zero_pad], 0)
        sub_sim.detach()
        sub_emb.detach()

        # absolute location features
        ann_boxes = yxyx_to_xyxy(np.vstack([box for box in self.Refs[ref_id]['bbxes']])[0:100, :])
        gd_box = self.Refs[ref_id]['bbxes'][-1]
        ann_boxes = np.vstack([ann_boxes, gd_box])
        lfeats = self.compute_lfeats(image_id, ann_boxes.tolist())
        lfeats = Variable(torch.from_numpy(lfeats))
        if lfeats.size(0) < 101:
            zero_pad = torch.zeros((101-lfeats.size(0)), 5)
            lfeats = torch.cat([lfeats, zero_pad], 0)     
        lfeats.detach()

        att_labels, select_ixs = self.fetch_attribute_label(ref_id)

        labels = self.fetch_seq(ref_id).value
        labels = Variable(torch.from_numpy(labels).long())

        return {
            "pool5_feats": pool5_feats,
            'fc7_feats'  : fc7_feats,
            'lfeats'     : lfeats,
            "labels"     : labels,
            "ref_id"     : ref_id,
            "image_id"   : image_id,
            "sub_sim"    : sub_sim,
            "sub_emb"    : sub_emb,
            'att_labels' : att_labels,
            'select_ixs' : select_ixs,
            'gd_boxes'   : torch.tensor(gd_boxes).float()
        }

    def __len__(self):
        return len(self.split_ix[self.split])

    def get_cats_weights(self, scale = 10):
        # weights = \lamda * 1/sqrt(cnt)
        cnts = [self.cat_to_cnt.get(cat, 1) for cat, ix in self.cat_to_ix.items()]
        cnts = np.array(cnts)
        weights = 1 / cnts ** 0.5
        weights = (weights-np.min(weights))/(np.max(weights)-np.min(weights))
        weights = weights * (scale - 1) + 1
        return torch.from_numpy(weights).float()
    
    def compute_lfeats(self, image_id, ann_boxs):
        # return ndarray float32 (#ann_ids, 5)
        lfeats = np.empty((len(ann_boxs), 5), dtype=np.float32)
        RootPath = '/home/xuejing_liu/yuki'
        img_path = osp.join(RootPath, 'data/Flickr30KEntities/flickr30k_images/flickr30k_images', image_id + '.jpg')
        im = skimage.io.imread(img_path)
        self.ih = im.shape[0]
        self.iw = im.shape[1]

        for ix, ann_box  in enumerate(ann_boxs):
            x1, y1, x2, y2 = ann_box[0], ann_box[1], ann_box[2], ann_box[3]
            lfeats[ix] = np.array([x1 / self.iw, y1 / self.ih, x2 / self.iw, y2 / self.ih, (x2-x1+1) * (y2-y1+1) / (self.iw * self.ih)], np.float32)
        return lfeats
        
#     def fetch_attribute_label(self, ref_ids):
#         """Return
#     - labels    : Variable float (N, num_atts)
#     - select_ixs: Variable long (n, )
#     """
#         labels = np.zeros((len(ref_ids), self.num_atts))
#         select_ixs = []
#         for i, ref_id in enumerate(ref_ids):
#             # pdb.set_trace()
#             ref = self.Refs[ref_id]
#             if len(ref['att_wds']) > 0:
#                 select_ixs += [i]
#                 for wd in ref['att_wds']:
#                     labels[i, self.att_to_ix[wd]] = 1

#         return Variable(torch.from_numpy(labels).float().cuda()), Variable(torch.LongTensor(select_ixs).cuda())

    def fetch_attribute_label(self, ref_id):
        """Return
    - labels    : Variable float (N, num_atts)
    - select_ixs: Variable long (n, )
    """
        labels = np.zeros(self.num_atts)
        ref = self.Refs[ref_id]
        select_ixs = [0]
        if len(ref['att_wds']) > 0:
            select_ixs = [1]
            for wd in ref['att_wds']:
                labels[self.att_to_ix[wd]] = 1

        return Variable(torch.from_numpy(labels).float()), Variable(torch.LongTensor(select_ixs))

    # 根据sent_id得到转化后的sentences
    def fetch_seq(self, sent_id):
        # return int32 (label_length, )
        ref_h5_id = self.Refs[sent_id]['h5_id']
        img_path = osp.join('/home/xuejing_liu/yuki/MattNet/cache/prepro/flickr30k/data_h5', str(ref_h5_id) + '.h5')
        self.data_h5 = h5py.File(img_path, 'r')
        seq = self.data_h5['labels']
        return seq

    # 将sentecne中的string通过vocab转化为id
    def encode_labels(self, sent_str_list):
        """Input:
        sent_str_list: list of n sents in string format
        return int32 (n, label_length) zeros padded in end
        """
        num_sents = len(sent_str_list)
        L = np.zeros((num_sents, self.label_length), dtype=np.int32)
        for i, sent_str in enumerate(sent_str_list):
            tokens = sent_str.split()
            for j, w in enumerate(tokens):
              if j < self.label_length:
                  L[i, j] = self.word_to_ix[w] if w in self.word_to_ix else self.word_to_ix['<UNK>']
        return L

    # 将sentence中的id转化为string
    def decode_labels(self, labels):
        """
        labels: int32 (n, label_length) zeros padded in end
        return: list of sents in string format
        """
        decoded_sent_strs = []
        num_sents = labels.shape[0]
        for i in range(num_sents):
            label = labels[i].tolist()
            sent_str = self.ix_to_word[label]
            decoded_sent_strs.append(sent_str)
        return decoded_sent_strs

    def get_attribute_weights(self, scale = 10):
        # weights = \lamda * 1/sqrt(cnt)
        cnts = [self.att_to_cnt[self.ix_to_att[ix]] for ix in range(self.num_atts)]
        cnts = np.array(cnts)
        weights = 1 / cnts ** 0.5
        weights = (weights-np.min(weights))/(np.max(weights)-np.min(weights))
        weights = weights * (scale - 1) + 1
        return torch.from_numpy(weights).float()



# # Original dataloader
# class Loader(object):

#     def __init__(self, data_json, similarity, data_h5=None, data_emb=None):
#         print('Loader loading similarity file:', similarity)
#         self.similarity_info = json.load(open(similarity))
#         self.similarity = self.similarity_info['sim']
#         print('Loader loading data.json: ', data_json)
#         self.info = json.load(open(data_json))
#         self.word_to_ix = self.info['word_to_ix']
#         self.ix_to_word = {ix: wd for wd, ix in self.word_to_ix.items()}
#         print('vocab size is ', self.vocab_size)
#         self.refs = self.info['refs']
#         print('we have %s refs.' % len(self.refs))
#         self.ix_to_cat = {}
#         self.cat_to_ix = {}
#         cls_wds = []
#         i = 0
#         for ref in self.refs:
#             if isinstance(ref['tokens'], str):
#                 cls_wd =ref['tokens'].split()[-1]
#             else:
#                 cls_wd = ref['tokens'][-1]
#             if cls_wd not in cls_wds:
#                 cls_wds.append(cls_wd)
#                 self.ix_to_cat[i]=cls_wd
#                 self.cat_to_ix[cls_wd]=i
#                 i += 1

#         print('object cateogry size is ', len(self.ix_to_cat))

#         print('label_length is ', self.label_length)

#         # construct mapping
#         self.Refs = {ref['ann_id']: ref for ref in self.refs}

#         # read data_h5 if exists
#         self.data_h5 = None
#         if data_h5 is not None:
#             print('Loader loading data.h5: ', data_h5)
#             self.data_h5 = h5py.File(data_h5, 'r')
#             assert self.data_h5['labels'].shape[0] == len(self.sentences), 'label.shape[0] not match sentences'
#             assert self.data_h5['labels'].shape[1] == self.label_length, 'label.shape[1] not match label_length'

#         self.data_emb = None
#         if data_emb is not None:
#             print('Loader loading data.h5: ', data_emb)
#             self.data_emb = h5py.File(data_emb, 'r')
#             assert self.data_emb['labels'].shape[0] == len(self.sentences), 'label.shape[0] not match sentences'
#             # assert self.data_emb['labels'].shape[1] == self.label_length, 'label.shape[1] not match label_length'

#         # self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

#     # @property装饰器负责把一个方法变成属性调用,广泛应用在类的定义中，可以让调用者写出简短的代码，同时保证对参数进行必要的检查
#     @property
#     def vocab_size(self):
#         # len(self.word_to_ix) == 1999
#         return len(self.word_to_ix)

#     @property
#     def label_length(self):
#         return self.info['label_length']

#     @property
#     def sent_to_Ref(self, sent_id):
#         return self.sent_to_Ref(sent_id)

#     # 将sentecne中的string通过vocab转化为id
#     def encode_labels(self, sent_str_list):
#         """Input:
#         sent_str_list: list of n sents in string format
#         return int32 (n, label_length) zeros padded in end
#         """
#         num_sents = len(sent_str_list)
#         L = np.zeros((num_sents, self.label_length), dtype=np.int32)
#         for i, sent_str in enumerate(sent_str_list):
#             tokens = sent_str.split()
#             for j, w in enumerate(tokens):
#               if j < self.label_length:
#                   L[i, j] = self.word_to_ix[w] if w in self.word_to_ix else self.word_to_ix['<UNK>']
#         return L

#     # 将sentence中的id转化为string
#     def decode_labels(self, labels):
#         """
#         labels: int32 (n, label_length) zeros padded in end
#         return: list of sents in string format
#         """
#         decoded_sent_strs = []
#         num_sents = labels.shape[0]
#         for i in range(num_sents):
#             label = labels[i].tolist()
#             sent_str = ' '.join([self.ix_to_word[int(i)] for i in label if i != 0])
#             decoded_sent_strs.append(sent_str)
#         return decoded_sent_strs

#     # 一个ref_id对应多个sent_id, 根据ref_id得到转化后的sentence
#     def fetch_label(self, ref_id, num_sents):
#         """
#         return: int32 (num_sents, label_length) and picked_sent_ids
#         """
#         ref = self.Refs[ref_id]
#         sent_ids = list(ref['sent_ids'])  # copy in case the raw list is changed
#         seq = []
#         # 将每个ref中的sent个数设置成一样的，如果个数少于num_sents,则将少的部分用已有的sent随机填充，否则，选择前num_sent个sentences
#         if len(sent_ids) < num_sents:
#             append_sent_ids = [random.choice(sent_ids) for _ in range(num_sents - len(sent_ids))]
#             sent_ids += append_sent_ids
#         else:
#             sent_ids = sent_ids[:num_sents]
#         assert len(sent_ids) == num_sents
#         # fetch label
#         for sent_id in sent_ids:
#             sent_h5_id = self.Sentences[sent_id]['h5_id']
#             seq += [self.data_h5['labels'][sent_h5_id, :]]
#         seq = np.vstack(seq)
#         return seq, sent_ids

#     # 根据sent_id得到转化后的sentences
#     def fetch_seq(self, sent_id):
#         # return int32 (label_length, )
#         ref_h5_id = self.Refs[sent_id]['h5_id']
#         img_path = osp.join('/home/xuejing_liu/yuki/MattNet/cache/prepro/flickr30k/data_h5', str(ref_h5_id) + '.h5')
#         self.data_h5 = h5py.File(img_path, 'r')
#         seq = self.data_h5['labels']
#         return seq
#     #
#     # def fetch_seq_bert(self, sent_id):
#     #     tokens = self.Sentences[sent_id]['tokens']
#     #     seq = self.tokenizer.convert_tokens_to_ids(tokens)
#     #     seq = np.array(seq)
#     #     return seq

#     def fetch_emb(self, sent_id):
#         # return int32 (label_length, )
#         sent_h5_id = self.Sentences[sent_id]['h5_id']
#         emb = self.data_emb['labels'][sent_h5_id, 0, :]
#         return emb

# class DataLoader(Loader):

#     def __init__(self, data_json, data_h5, similarity, opt):
#         # parent loader instance
#         Loader.__init__(self, data_json, similarity, data_h5)
#         # prepare attributes
#         self.att_to_ix = self.info['att_to_ix']
#         self.ix_to_att = {ix: wd for wd, ix in self.att_to_ix.items()}
#         self.num_atts = len(self.att_to_ix)
#         self.att_to_cnt = self.info['att_to_cnt']
#         self.vocab_file = '/home/xuejing_liu/yuki/MattNet/cache/word_embedding/vocabulary_72700.txt'
#         self.UNK_IDENTIFIER = '<unk>'  # <unk> is the word used to identify unknown words
#         self.num_vocab = 72704
#         self.embed_dim = 300
#         self.embedmat_path = '/home/xuejing_liu/yuki/MattNet/cache/word_embedding/embed_matrix.npy'
#         self.embedding_mat = np.load(self.embedmat_path)

#         # img_iterators for each split
#         self.split_ix = {}
#         self.iterators = {}
#         for ref_id, ref in self.Refs.items():
#             # we use its ref's split (there is assumption that each image only has one split)
#             split = ref['split']
#             if split not in self.split_ix:
#                 self.split_ix[split] = []
#                 self.iterators[split] = 0
#             self.split_ix[split] += [ref_id]
#         for k, v in self.split_ix.items():
#             print('assigned %d images to split %s' %(len(v), k))

#     # load vocabulary file
#     def load_vocab_dict_from_file(self, dict_file):
#         if (sys.version_info > (3, 0)):
#             with open(dict_file, encoding='utf-8') as f:
#                 words = [w.strip() for w in f.readlines()]
#         else:
#             with io.open(dict_file, encoding='utf-8') as f:
#                 words = [w.strip() for w in f.readlines()]
#         vocab_dict = {words[n]: n for n in range(len(words))}
#         return vocab_dict

#     # get word location in vocabulary
#     def words2vocab_indices(self, words, vocab_dict):
#         if isinstance(words, str):
#             vocab_indices = [vocab_dict[words] if words in vocab_dict else vocab_dict[self.UNK_IDENTIFIER]]
#         else:
#             vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[self.UNK_IDENTIFIER])
#                              for w in words]
#         return vocab_indices

#     # get ann category
#     def get_ann_category(self, image_id):
#         image = self.Images[image_id]
#         ann_ids = image['ann_ids']
#         vocab_dict = self.load_vocab_dict_from_file(self.vocab_file)
#         ann_cats = []

#         for ann_id in ann_ids:
#             ann = self.Anns[ann_id]
#             cat_id = ann['category_id']
#             cat = self.ix_to_cat[cat_id]
#             cat = cat.split(' ')
#             cat = self.words2vocab_indices(cat, vocab_dict)
#             ann_cats += [cat]
#         return ann_cats

#     # get category for each sentence
#     def get_sent_category(self, image_id):
#         image = self.Images[image_id]
#         ref_ids = image['ref_ids']
#         vocab_dict = self.load_vocab_dict_from_file(self.vocab_file)
#         sub_wds = []
#         obj_wds = []
#         loc_wds = []
#         rel_loc_wds = []
#         for ref_id in ref_ids:
#             ref = self.Refs[ref_id]
#             sent_ids = ref['sent_ids']
#             for sent_id in sent_ids:
#                 att_wds = self.sub_obj_wds[str(sent_id)]
#                 sub_wd = att_wds['r1']
#                 sub_wd = self.words2vocab_indices(sub_wd, vocab_dict)
#                 obj_wd = att_wds['r6']
#                 obj_wd = self.words2vocab_indices(obj_wd, vocab_dict)
#                 sub_wds += [sub_wd]
#                 obj_wds += [obj_wd]
#         none_idx = self.words2vocab_indices('none', vocab_dict)


#         sent_att = {}
#         sent_att['sub_wds'] = sub_wds
#         sent_att['obj_wds'] = obj_wds
#         return sent_att, none_idx

#     def prepare_mrcn(self, head_feats_dir, args):
#         """
#         Arguments:
#           head_feats_dir: cache/feats/dataset_splitBy/net_imdb_tag, containing all image conv_net feats
#           args: imdb_name, net_name, iters, tag
#         """
#         self.head_feats_dir = head_feats_dir
#         self.mrcn = inference_no_imdb.Inference(args)
#         assert args.net_name == 'res101'
#         self.pool5_dim = 1024
#         self.fc7_dim = 2048

#     # load different kinds of feats
#     def loadFeats(self, Feats):
#         # Feats = {feats_name: feats_path}
#         self.feats = {}
#         self.feat_dim = None
#         for feats_name, feats_path in Feats.items():
#             if osp.isfile(feats_path):
#                 self.feats[feats_name] = h5py.File(feats_path, 'r')
#                 self.feat_dim = self.feats[feats_name]['fc7'].shape[1]
#                 assert self.feat_dim == self.fc7_dim
#                 print('FeatLoader loading [%s] from %s [feat_dim %s]' %(feats_name, feats_path, self.feat_dim))

#     # shuffle split
#     def shuffle(self, split):
#         random.shuffle(self.split_ix[split])

#     # reset iterator
#     def resetIterator(self, split):
#         self.iterators[split]=0

#     # expand list by seq per ref, i.e., [a,b], 3 -> [aaabbb]
#     def expand_list(self, L, n):
#         out = []
#         for l in L:
#             out += [l] * n
#         return out

#     def image_to_head(self, image_id):
#         """Returns
#         head: float32 (1, 1024, H, W)
#         im_info: float32 [[im_h, im_w, im_scale]]
#         """
#         feats_h5 = osp.join(self.head_feats_dir, str(image_id)+'.h5')
#         feats = h5py.File(feats_h5, 'r')
#         head, im_info = feats['head'], feats['im_info']
#         return np.array(head), np.array(im_info)

#     def fetch_grid_feats(self, boxes, net_conv, im_info):
#         """returns -pool5 (n, 1024, 7, 7) -fc7 (n, 2048, 7, 7)"""
#         pool5, fc7 = self.mrcn.box_to_spatial_fc7(net_conv, im_info, boxes)
#         return pool5, fc7

#     def compute_lfeats(self, image_id, ann_boxs):
#         # return ndarray float32 (#ann_ids, 5)
#         lfeats = np.empty((len(ann_boxs), 5), dtype=np.float32)
#         RootPath = '/home/xuejing_liu/yuki'
#         img_path = osp.join(RootPath, 'data/Flickr30KEntities/flickr30k_images/flickr30k_images', image_id + '.jpg')
#         im = skimage.io.imread(img_path)
#         self.ih = im.shape[0]
#         self.iw = im.shape[1]

#         for ix, ann_box  in enumerate(ann_boxs):
#             x1, y1, x2, y2 = ann_box[0], ann_box[1], ann_box[2], ann_box[3]
#             lfeats[ix] = np.array([x1 / self.iw, y1 / self.ih, x2 / self.iw, y2 / self.ih, (x2-x1+1) * (y2-y1+1) / (self.iw * self.ih)], np.float32)
#         return lfeats


#     def fetch_attribute_label(self, ref_ids):
#         """Return
#     - labels    : Variable float (N, num_atts)
#     - select_ixs: Variable long (n, )
#     """
#         labels = np.zeros((len(ref_ids), self.num_atts))
#         select_ixs = []
#         for i, ref_id in enumerate(ref_ids):
#             # pdb.set_trace()
#             ref = self.Refs[ref_id]
#             if len(ref['att_wds']) > 0:
#                 select_ixs += [i]
#                 for wd in ref['att_wds']:
#                     labels[i, self.att_to_ix[wd]] = 1

#         return Variable(torch.from_numpy(labels).float().cuda()), Variable(torch.LongTensor(select_ixs).cuda())

#     def fetch_cxt_feats(self, ann_boxes, opt, obj_sim):
#         """
#         Return
#         - cxt_feats : ndarray (#ann_ids, fc7_dim)
#         - cxt_lfeats: ndarray (#ann_ids, ann_ids, 5)
#         - dist: ndarray (#ann_ids, ann_ids, 1)
#         Note we only use neighbouring "different" (+ "same") objects for computing context objects, zeros padded.
#         """

#         # cxt_feats = np.zeros((len(ann_boxes), self.fc7_dim), dtype=np.float32)
#         cxt_lfeats = np.zeros((len(ann_boxes), len(ann_boxes), 5), dtype=np.float32)
#         dist = np.zeros((len(ann_boxes), len(ann_boxes), 1), dtype=np.float32)

#         for i, ann_box in enumerate(ann_boxes):
#             # reference box
#             rbox = ann_box
#             rcx, rcy, rw, rh = (rbox[0]+rbox[2])/2, (rbox[1]+rbox[3])/2, (rbox[2]-rbox[0]+1), (rbox[3]-rbox[1]+1)
#             for j, cand_ann_box in enumerate(ann_boxes):
#                 # cand_ann = self.Anns[cand_ann_id]
#                 # fc7_feats
#                 # cxt_feats[i, :] = self.feats['ann']['fc7'][cand_ann['h5_id'], :]
#                 cbox = cand_ann_box
#                 cx1, cy1, cw, ch = cbox[0], cbox[1], (cbox[2]-cbox[0]+1), (cbox[3]-cbox[1]+1)
#                 ccx, ccy = (cbox[0]+cbox[2])/2, (cbox[1]+cbox[3])/2
#                 dist[i,j,:] = np.array(abs(rcx-ccx)+abs(rcy-ccy))
#                 cxt_lfeats[i,j,:] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
#         return cxt_lfeats, dist
#         #return cxt_feats, cxt_lfeats, dist

#     # get batch of data
#     def getBatch(self, split, opt):
#         # options
#         batch_size = opt.get('batch_size', 5)
#         split_ix = self.split_ix[split]
#         max_index = len(split_ix) - 1
#         wrapped = False

#         batch_ref_ids = []
#         for i in range(batch_size):
#             ri = self.iterators[split]
#             ri_next = ri + 1
#             if ri_next > max_index:
#                 ri_next = 0
#                 wrapped = True
#             self.iterators[split] = ri_next
#             ref_id = split_ix[ri]
#             batch_ref_ids += [ref_id]

#         batch_image_ids = []
#         batch_fc7 = []
#         batch_pool5 = []
#         batch_lfeats = []
#         batch_sub_sim, batch_sub_emb = [], []

#         for ref_id in batch_ref_ids:
#             image_id = self.Refs[ref_id]['image_id']
#             batch_image_ids += [image_id]

#             head, im_info = self.image_to_head(image_id)
#             head = Variable(torch.from_numpy(head).cuda())

#             sub_sim = torch.Tensor(self.similarity[str(ref_id)]['sub_sim']).cuda().squeeze(2)
#             sub_emb = torch.Tensor(self.similarity[str(ref_id)]['sub_emb']).cuda()
 
#             # fetch ann features
#             ann_boxes = np.vstack([box for box in self.Refs[ref_id]['bbxes']])[0:100, :]
#             ann_pool5, ann_fc7 = self.fetch_grid_feats(ann_boxes, head, im_info)

#             if sub_sim.size(1) < 100:
#                 zero_pad = torch.zeros(1, (100-sub_sim.size(1))).cuda()
#                 sub_sim = torch.cat([sub_sim, zero_pad], 1)

#             if ann_pool5.size(0) < 100:
#                 zero_pad = torch.zeros((100-ann_pool5.size(0)), 1024, 7, 7).cuda()
#                 ann_pool5 = torch.cat([ann_pool5, zero_pad], 0)

#             if ann_fc7.size(0) < 100:
#                 zero_pad = torch.zeros((100-ann_fc7.size(0)), 2048, 7, 7).cuda()
#                 ann_fc7 = torch.cat([ann_fc7, zero_pad], 0)

#             ann_pool5 = ann_pool5.unsqueeze(0)
#             ann_fc7 = ann_fc7.unsqueeze(0)

#             batch_sub_sim += [sub_sim]
#             batch_sub_emb += [sub_emb]
#             batch_fc7 += [ann_fc7]
#             batch_pool5 += [ann_pool5]

#             # absolute location features
#             lfeats = self.compute_lfeats(image_id, self.Refs[ref_id]['bbxes'][0:100])
#             lfeats = Variable(torch.from_numpy(lfeats).cuda())
#             if lfeats.size(0) < 100:
#                 zero_pad = torch.zeros((100-lfeats.size(0)), 5).cuda()
#                 lfeats = torch.cat([lfeats, zero_pad], 0)            
#             batch_lfeats += [lfeats.unsqueeze(0)]

#         fc7 = torch.cat(batch_fc7, 0)
#         fc7.detach()
#         pool5 = torch.cat(batch_pool5, 0)
#         pool5.detach()
#         lfeats = torch.cat(batch_lfeats, 0)
#         lfeats.detach()

#         sub_sim = torch.cat(batch_sub_sim, 0)
#         sub_sim.detach()
#         sub_emb = torch.cat(batch_sub_emb, 0)
#         sub_emb.detach()

#         att_labels, select_ixs = self.fetch_attribute_label(batch_ref_ids)

#         labels =  np.vstack([self.fetch_seq(ref_id).value for ref_id in batch_ref_ids]) 
#         labels = Variable(torch.from_numpy(labels).long().cuda())
#         max_len = (labels != 0).sum(1).max().data[0]
#         labels = labels[:, :max_len]

#         start_words = np.ones([labels.size(0), 1], dtype=int) * (self.word_to_ix['<s>'])
#         start_words = Variable(torch.from_numpy(start_words).long().cuda())
#         enc_labels = labels.clone()
#         enc_labels = torch.cat([start_words, enc_labels], 1)

#         zero_pad = np.zeros([labels.size(0), 1], dtype=int)
#         zero_pad = Variable(torch.from_numpy(zero_pad).long().cuda())
#         dec_labels = labels.clone()
#         dec_labels = torch.cat([dec_labels, zero_pad], 1)

#         # pdb.set_trace()

#         data = {}
#         data['labels'] = labels
#         data['enc_labels'] = enc_labels
#         data['dec_labels'] = dec_labels
#         data['ref_ids'] = batch_ref_ids
#         data['image_id'] = batch_image_ids
#         data['sim'] = {'sub_sim': sub_sim, 'sub_emb': sub_emb}
#         data['Feats'] = {'fc7': fc7, 'pool5': pool5, 'lfeats': lfeats}
#         data['att_labels'] = att_labels
#         data['select_ixs'] = select_ixs
#         data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}
#         return data

#     def get_attribute_weights(self, scale = 10):
#         # weights = \lamda * 1/sqrt(cnt)
#         cnts = [self.att_to_cnt[self.ix_to_att[ix]] for ix in range(self.num_atts)]
#         cnts = np.array(cnts)
#         weights = 1 / cnts ** 0.5
#         weights = (weights-np.min(weights))/(np.max(weights)-np.min(weights))
#         weights = weights * (scale - 1) + 1
#         return torch.from_numpy(weights).float()

#     def decode_attribute_label(self, scores):
#         """- scores: Variable (cuda) (n, num_atts) after sigmoid range [0, 1]
#            - labels:list of [[att, sc], [att, sc], ...
#         """
#         scores = scores.data.cpu().numpy()
#         N = scores.shape[0]
#         labels = []
#         for i in range(N):
#             label = []
#             score = scores[i]
#             for j, sc in enumerate(list(score)):
#                 label += [(self.ix_to_att[j], sc)]
#                 labels.append(label)
#         return labels

#     def getTestBatch(self, split, opt):

#         wrapped = False
#         split_ix = self.split_ix[split]
#         max_index = len(split_ix) - 1

#         ri = self.iterators[split]
#         ri_next = ri + 1
#         if ri_next > max_index:
#             ri_next = 0
#             wrapped = True
#         self.iterators[split] = ri_next

#         ref_id = split_ix[ri]
#         image_id = self.Refs[ref_id]['image_id']

#         ref = self.Refs[ref_id]

#         head, im_info = self.image_to_head(image_id)
#         head = Variable(torch.from_numpy(head).cuda())

#         # fetch ann features
#         # ann_boxes = np.vstack([box for box in self.Refs[ref_id]['bbxes']])
#         ann_boxes = np.vstack([box for box in self.Refs[ref_id]['bbxes'][0:100]])
#         ann_pool5, ann_fc7 = self.fetch_grid_feats(ann_boxes, head, im_info)

#         pool5 = ann_pool5.unsqueeze(0)
#         pool5 = pool5.detach()
#         fc7 = ann_fc7.unsqueeze(0)
#         fc7 = fc7.detach()

#         lfeats = self.compute_lfeats(image_id, self.Refs[ref_id]['bbxes'][0:100])
#         lfeats = Variable(torch.from_numpy(lfeats).cuda())
#         lfeats = lfeats.unsqueeze(0)
#         lfeats.detach()

#         sub_sim = torch.Tensor(self.similarity[str(ref_id)]['sub_sim']).cuda().squeeze(2)
#         sub_emb = torch.Tensor(self.similarity[str(ref_id)]['sub_emb']).cuda()

#         sub_emb.detach()
#         sub_sim.detach()

#         gd_boxes = ref['box']

#         att_labels, select_ixs = self.fetch_attribute_label([ref_id])

#         labels = self.fetch_seq(ref_id).value
#         labels = Variable(torch.from_numpy(labels).long().cuda())
#         labels = labels.unsqueeze(0)
#         max_len = (labels != 0).sum(1).max().data[0]
#         labels = labels[:, :max_len]

#         start_words = np.ones([labels.size(0), 1], dtype=int) * (self.word_to_ix['<s>'])
#         start_words = Variable(torch.from_numpy(start_words).long().cuda())
#         enc_labels = labels.clone()
#         enc_labels = torch.cat([start_words, enc_labels], 1)

#         zero_pad = np.zeros([labels.size(0), 1], dtype=int)
#         zero_pad = Variable(torch.from_numpy(zero_pad).long().cuda())
#         dec_labels = labels.clone()
#         dec_labels = torch.cat([dec_labels, zero_pad], 1)

#         data = {}
#         data['labels'] = labels
#         data['enc_labels'] = enc_labels
#         data['dec_labels'] = dec_labels
#         data['ref_ids'] = ref_id
#         data['image_id'] = image_id
#         data['gd_boxes'] = gd_boxes
#         data['sim'] = {'sub_sim': sub_sim, 'sub_emb': sub_emb}
#         data['Feats'] = {'fc7': fc7, 'pool5': pool5, 'lfeats': lfeats}
#         data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}
#         data['att_labels'] = att_labels
#         data['select_ixs'] = select_ixs

#         return data