import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon, Rectangle
import skimage.io as io
import os.path as osp
import os
import random
import seaborn as sns
import numpy as np
import argparse

def load_file(dataset,split_by,exp_id,test_set):
    #原始数据文件
    dataset_splitby = dataset+'_'+split_by
    info = json.load(open(osp.join('/home/xuejing_liu/yuki/MattNet/cache/prepro', dataset_splitby, 'data.json')))
    images = info['images']
    anns = info['anns']
    Images = {image['image_id']: image for image in images}
    Anns = {ann['ann_id']: ann for ann in anns}

    #获取图片名称
    instances = json.load(open(osp.join('/home/xuejing_liu/yuki/MattNet/data',dataset,'instances.json'), 'r'))
    imgs = instances['images']
    Imgs = {}
    for img in imgs:
        Imgs[img['id']] = img

    #结果文件   
    data_flie = osp.join('/home/xuejing_liu/yuki/KARN/result', dataset_splitby,exp_id+'_'+test_set+'.json')
    data = json.load(open(data_flie))
    predictions = data['predictions']
    return  Images, Anns, Imgs, predictions


def save_result(img_path, gd_ann_box, pred_ann_box, data_set, rorw, sent_id):
    plt.figure()
    ax = plt.gca()
    I = io.imread(img_path)
    ax.imshow(I)
    gd_box_plot = Rectangle((gd_ann_box[0], gd_ann_box[1]), gd_ann_box[2], gd_ann_box[3], fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(gd_box_plot)
    pred_box_plot = Rectangle((pred_ann_box[0], pred_ann_box[1]), pred_ann_box[2], pred_ann_box[3], fill=False, edgecolor='blue', linewidth=3, linestyle='--')
    ax.add_patch(pred_box_plot)
    plt.axis('off')
    # save
    save_dir = osp.join("/home/xuejing_liu/yuki/visualization", data_set, rorw,'result')
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join("/home/xuejing_liu/yuki/visualization", data_set, rorw,'result',str(sent_id)+'.png')
    plt.savefig(save_path)
    plt.close()

#可视化语言注意力
def save_lan_attn(sent, sub_attn_lan, loc_attn_lan, rel_attn_lan, weights, data_set, rorw, sent_id):
    plt.figure()
    x_label = [word for word in sent.split(" ")]
    sub_lan = np.array([sub_attn_lan])
    loc_lan = np.array([loc_attn_lan])
    rel_lan = np.array([rel_attn_lan])
    lan_attn = np.vstack((sub_lan, loc_lan, rel_lan))
    sns.heatmap(lan_attn, fmt=".2g", xticklabels=x_label, yticklabels=['sub '+str(round(weights[0],2)), 'loc '+str(round(weights[1],2)), 'cxt '+str(round(weights[2],2))],robust=False, cmap='YlGnBu')
    # save
    save_dir = osp.join("/home/xuejing_liu/yuki/visualization", data_set, rorw,'lan_attn')
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join("/home/xuejing_liu/yuki/visualization", data_set, rorw,'lan_attn',str(sent_id)+'.png')
    plt.savefig(save_path)
    plt.close()

def save_sub_attn(img_path, Images, Anns, image_id, sub_attn, data_set, rorw, sent_id):
    plt.figure()
    ax = plt.gca()
    I = io.imread(img_path)
    ax.imshow(I)
    ann_ids = Images[image_id]['ann_ids']
    for i, ann_id in enumerate(ann_ids):
        ann_attn = round(np.clip(sub_attn[i],0,1), 2)
        ann_box = Anns[ann_id]['box']
        box_plot = Rectangle((ann_box[0], ann_box[1]), ann_box[2], ann_box[3], fill=True, facecolor='blue', alpha=ann_attn, edgecolor='blue', linewidth=2)
        ax.add_patch(box_plot)
        ax.text(ann_box[0], ann_box[1], str(ann_attn), fontsize=10, color='red')
    plt.axis('off')
    save_dir = osp.join("/home/xuejing_liu/yuki/visualization", data_set, rorw,'sub_attn')
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join("/home/xuejing_liu/yuki/visualization", data_set, rorw,'sub_attn',str(sent_id)+'.png')
    plt.savefig(save_path)
    plt.close()

def save_obj_attn(img_path, Images, Anns, image_id, obj_attn, data_set, rorw, sent_id):
    plt.figure()
    ax = plt.gca()
    I = io.imread(img_path)
    ax.imshow(I)
    ann_ids = Images[image_id]['ann_ids']
    for i, ann_id in enumerate(ann_ids):
        ann_attn = round(np.clip(obj_attn[i],0,1), 2)
        ann_box = Anns[ann_id]['box']
        box_plot = Rectangle((ann_box[0], ann_box[1]), ann_box[2], ann_box[3], fill=True, facecolor='blue', alpha=ann_attn, edgecolor='red', linewidth=3)
        ax.add_patch(box_plot)
        ax.text(ann_box[0], ann_box[1], str(ann_attn), fontsize=10, color='red')
    plt.axis('off')
    save_dir = osp.join("/home/xuejing_liu/yuki/visualization", data_set, rorw,'obj_attn')
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join("/home/xuejing_liu/yuki/visualization", data_set, rorw,'obj_attn',str(sent_id)+'.png')
    plt.savefig(save_path)
    plt.close()


def save_fnl_sub_attn(img_path, Images, Anns, image_id, sub_ann_attn, data_set, rorw, sent_id):
    plt.figure()
    ax = plt.gca()
    I = io.imread(img_path)
    ax.imshow(I)
    ann_ids = Images[image_id]['ann_ids']
    for i, ann_id in enumerate(ann_ids):
        ann_attn = round(np.clip(sub_ann_attn[i][0],0,1), 2)
        ann_box = Anns[ann_id]['box']
        box_plot = Rectangle((ann_box[0], ann_box[1]), ann_box[2], ann_box[3], fill=True, facecolor='blue', alpha=ann_attn, edgecolor='blue', linewidth=3)
        ax.add_patch(box_plot)
        ax.text(ann_box[0], ann_box[1], str(ann_attn), fontsize=10, color='red')
    plt.axis('off')
    save_dir = osp.join("/home/xuejing_liu/yuki/visualization", data_set, rorw,'fnl_sub_attn')
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join("/home/xuejing_liu/yuki/visualization", data_set, rorw,'fnl_sub_attn',str(sent_id)+'.png')
    plt.savefig(save_path)
    plt.close()

def save_fnl_loc_attn(img_path, Images, Anns, image_id, loc_ann_attn, data_set, rorw, sent_id):
    plt.figure()
    ax = plt.gca()
    I = io.imread(img_path)
    ax.imshow(I)
    ann_ids = Images[image_id]['ann_ids']
    for i, ann_id in enumerate(ann_ids):
        ann_attn = round(np.clip(loc_ann_attn[i][0],0,1), 2)
        ann_box = Anns[ann_id]['box']
        box_plot = Rectangle((ann_box[0], ann_box[1]), ann_box[2], ann_box[3], fill=True, facecolor='blue', alpha=ann_attn, edgecolor='blue', linewidth=3)
        ax.add_patch(box_plot)
        ax.text(ann_box[0], ann_box[1], str(ann_attn), fontsize=10, color='red')
    plt.axis('off')
    save_dir = osp.join("/home/xuejing_liu/yuki/visualization", data_set, rorw,'fnl_loc_attn')
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join("/home/xuejing_liu/yuki/visualization", data_set, rorw,'fnl_loc_attn',str(sent_id)+'.png')
    plt.savefig(save_path)
    plt.close()

def save_fnl_rel_attn(img_path, Images, Anns, image_id, rel_ann_attn, data_set, rorw, sent_id):
    plt.figure()
    ax = plt.gca()
    I = io.imread(img_path)
    ax.imshow(I)
    ann_ids = Images[image_id]['ann_ids']
    for i, ann_id in enumerate(ann_ids):
        ann_attn = round(np.clip(rel_ann_attn[i][0],0,1), 2)
        ann_box = Anns[ann_id]['box']
        box_plot = Rectangle((ann_box[0], ann_box[1]), ann_box[2], ann_box[3], fill=True, facecolor='blue', alpha=ann_attn, edgecolor='blue', linewidth=3)
        ax.add_patch(box_plot)
        ax.text(ann_box[0], ann_box[1], str(ann_attn), fontsize=10, color='red')
    plt.axis('off')
    save_dir = osp.join("/home/xuejing_liu/yuki/visualization", data_set, rorw,'fnl_rel_attn')
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join("/home/xuejing_liu/yuki/visualization", data_set, rorw,'fnl_rel_attn',str(sent_id)+'.png')
    plt.savefig(save_path)
    plt.close()

def visualization(params):       
    dataset = params['data']
    split_by = params['splitBy']
    exp_id = params['id']
    test_set = params['split']
    data_set = params['data'] + params['split']
    Images, Anns, Imgs, predictions = load_file(dataset,split_by,exp_id,test_set)
    IMAGE_DIR = '/home/xuejing_liu/yuki/MattNet/data/images/mscoco/images/train2014'
    length = len(predictions)
    ids = random.sample(range(length),250)
    right_num = 0
    save_num = 0
    for id in ids:
        result = predictions[id]
        image_id = result['image_id']
        sent_id = result['sent_id']
        sent = result['sent']
        gd_ann_id = result['gd_ann_id']
        gd_ann_box = Anns[gd_ann_id]['box']
        pred_ann_id = result['pred_ann_id']
        pred_ann_box = Anns[pred_ann_id]['box']
        pred_score = result['pred_score']
        IoU = result['IoU']
        sub_attn = result['sub_attn']
        obj_attn = result['obj_attn']
        weights = result['weights']
        sub_attn_lan = result['sub_attn_lan']
        loc_attn_lan = result['loc_attn_lan'] 
        rel_attn_lan = result['rel_attn_lan']
        sub_ann_attn = result['sub_ann_attn']
        loc_ann_attn = result['loc_ann_attn']
        rel_ann_attn = result['rel_ann_attn']
        img_path = osp.join(IMAGE_DIR, Imgs[image_id]['file_name'])
        if IoU >= 0.5:
            save_result(img_path, gd_ann_box, pred_ann_box, data_set, 'right', sent_id)
            save_lan_attn(sent, sub_attn_lan, loc_attn_lan, rel_attn_lan, weights, data_set, 'right', sent_id)
            save_sub_attn(img_path, Images, Anns, image_id, sub_attn, data_set, 'right', sent_id)
            save_obj_attn(img_path, Images, Anns, image_id, obj_attn, data_set, 'right', sent_id)
            save_fnl_sub_attn(img_path, Images, Anns, image_id, sub_ann_attn, data_set, 'right', sent_id)
            save_fnl_loc_attn(img_path, Images, Anns, image_id, loc_ann_attn, data_set, 'right', sent_id)
            save_fnl_rel_attn(img_path, Images, Anns, image_id, rel_ann_attn, data_set, 'right', sent_id)
            right_num += 1
            if right_num > 100:
                break
        else:
            save_result(img_path, gd_ann_box, pred_ann_box, data_set, 'wrong', sent_id)
            save_lan_attn(sent, sub_attn_lan, loc_attn_lan, rel_attn_lan, weights, data_set, 'wrong', sent_id)
            save_sub_attn(img_path, Images, Anns, image_id, sub_attn, data_set, 'wrong', sent_id)
            save_obj_attn(img_path, Images, Anns, image_id, obj_attn, data_set, 'wrong', sent_id)
            save_fnl_sub_attn(img_path, Images, Anns, image_id, sub_ann_attn, data_set, 'wrong', sent_id)
            save_fnl_loc_attn(img_path, Images, Anns, image_id, loc_ann_attn, data_set, 'wrong', sent_id)
            save_fnl_rel_attn(img_path, Images, Anns, image_id, rel_ann_attn, data_set, 'wrong', sent_id)
        save_num += 1
        print("save image %s" %(str(image_id)))
    print('%i images saved!' %(save_num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='refcoco',
                        help='dataset name: refclef, refcoco, refcoco+, refcocog')
    parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
    parser.add_argument('--split', type=str, default='val', help='split: testAB or val, etc')
    parser.add_argument('--id', type=str, default='exp6', help='model id name')
    args = parser.parse_args()
    params = vars(args)

    visualization(params)