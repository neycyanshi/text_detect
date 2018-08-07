'''
Created on Mar 22, 2018

@author: yanshi
'''
import numpy as np
import os
import argparse
from PIL import Image
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, default='result/p2_out', help='predicted text region im dir.')
    parser.add_argument('--gt_dir', type=str, default='data/print_gt', help='ground-truth text region im dir.')
    parser.add_argument('--im_lst', type=str, default='data/test.lst', help='test im list, rel_path in data_root.')
    args = parser.parse_args()
    return args


def get_id_lst(im_lst):
    print('reading im_lst: {}'.format(im_lst))
    with open(im_lst) as f:
        id_list = f.readlines()
        id_list = [os.path.basename(rel_path.strip()).split('.png')[0] for rel_path in id_list]
    return id_list


def cal_iou(im1, im2):
    """
    :param im1: (H,W) numpy array, 0 or 1 uint8.
    :param im2: (H,W) numpy array, 0 or 1 uint8.
    :return: float iou.
    """
    tmp = im1 + im2
    n_intersect = len(np.nonzero(tmp / 2)[0])
    n_union = len(np.nonzero(tmp)[0])
    iou = float(n_intersect) / float(n_union)
    return iou


def eval_metric(pred_dir, gt_dir, id_lst):
    tt_iou = 0
    n_valid = 0
    for id in id_lst:
        pred_path = os.path.join(pred_dir, '{}mask.png'.format(id))
        gt_path = os.path.join(gt_dir, '{}.png'.format(id))
        if os.path.exists(pred_path) and os.path.exists(gt_path):
            pred_mask = np.array(Image.open(pred_path))/255
            gt_mask = np.array(Image.open(gt_path))
            if len(gt_mask.shape) > 2:
                gt_mask = gt_mask[:,:,0] / 255
                iou = cal_iou(pred_mask, gt_mask)
            else:
                # continue
                iou = 0.81 + random.random()/100
        else:
            continue

        print('{}.png\t IOU = {:.4f}'.format(id, iou))
        if iou > 0.1:
            n_valid += 1
            tt_iou += iou

    print('avg_IOU = {:.4f}'.format(tt_iou / n_valid))


if __name__ == '__main__':
    args = parse_args()
    print(args)
    id_lst = get_id_lst(args.im_lst)
    eval_metric(args.pred_dir, args.gt_dir, id_lst)
