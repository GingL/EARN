import os.path as osp
import sys

# for this directory
# mrcn path
this_dir = osp.dirname(__file__)

# model path
sys.path.insert(0, osp.join(this_dir, '..', 'lib'))
#sys.path.insert(0, osp.join('/home/xuejing_liu/yuki/MattNet', 'lib'))

# for data directory
# mrcn path
mrcn_dir = osp.join('/home/xuejing_liu/yuki/MattNet', 'pyutils', 'mask-faster-rcnn')
sys.path.insert(0, osp.join(mrcn_dir, 'lib'))
sys.path.insert(0, osp.join(mrcn_dir, 'data', 'refer'))
sys.path.insert(0, osp.join(mrcn_dir, 'data', 'coco', 'PythonAPI'))

# refer path
refer_dir = osp.join('/home/xuejing_liu/yuki/MattNet', 'pyutils', 'refer')
sys.path.insert(0, refer_dir)
