from __future__ import division
import numpy as np
from PIL import Image
import os
import cv2
import time
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--img_name', type=str, default=None, help='test img name')
parser.add_argument('--data_root', type=str, default='ori', help='path to data root folder.')
parser.add_argument('--dest_folder', type=str, default='p1_out', help='path to phase 1 save side-output and fused result.')
parser.add_argument('--model_root', type=str, default='model', help='root for deploy.txt and caffemodel.')
parser.add_argument('--caffe_root', type=str, default='~/caffe', help='root path for your caffe.')
parser.add_argument('--vis', type=bool, default='True', help='show input and output while processing.')

args = parser.parse_args()
print(args)

if not os.path.exists(args.dest_folder):
    os.makedirs(args.dest_folder)
    print('making new dest_folder: {}'.format(args.dest_folder))

caffe_root = args.caffe_root
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

def plot_single_scale(scale_lst, name):
    for i in range(0, len(scale_lst)):
        image = Image.fromarray(255.0*(scale_lst[i])).convert('L')
        img_path = os.path.join(args.dest_folder, str(name) + str(i) + '.png')
        print('img saving to: {}'.format(img_path))
        image.save(img_path,'png')

data_root = args.data_root
if args.img_name is not None:
    test_lst = [os.path.join(data_root, args.img_name)]
else:
    test_lst = glob(data_root+'*.png')
name_lst = [os.path.basename(x).split('.')[0] for x in test_lst]
#print('test_lst: {}'.format(test_lst))
#print('name_lst: {}'.format(name_lst))

im_lst = []
print('total {} images to process.'.format(len(test_lst)))
for i in range(0, len(test_lst)):
    im = Image.open(test_lst[i])
    #if sz[0] > sz[1]:
    #        im = im.resize((960,int(sz[1]*960/sz[0])),Image.ANTIALIAS)
    #    else:
    #        im = im.resize((int(sz[0]*960/sz[1]),960),Image.ANTIALIAS)
    # sz = im.size
    # if sz[0]*sz[1] > 2500000:
    #     im = im.resize((int(sz[0]/2), int(sz[1]/2)), Image.ANTIALIAS)
    im = np.array(im, dtype=np.float32)
    print('{}: shape: {}'.format(test_lst[i], im.shape))
    im = im[:,:,::-1]
    # in_gray = np.dot(in_, [0.299, 0.587, 0.114]) / 255.0
    # print(in_gray.shape); print(in_gray)
    # plot_single_scale([in_gray],str(name_lst[i])+'in')
    im_lst.append(im)

caffe.set_mode_gpu()
caffe.set_device(0)

model_root = args.model_root
net_path = os.path.join(model_root, 'deploy.prototxt')
weight_path = os.path.join(model_root, 'hed_pretrained_bsds.caffemodel')
net = caffe.Net(net_path, weight_path, caffe.TEST)

# starttime = time.time()
vis_time = 10
for idx in range(0, len(im_lst)):
    print('{} | processing: {}'.format(idx, test_lst[idx]))
    in_ = im_lst[idx]
    cv2.imshow('input', in_.astype(np.uint8))
    cv2.moveWindow('input', 0, 0)
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    net.blobs['data'].reshape(1,*in_.shape)
    net.blobs['data'].data[...] = in_
    print('='*10 + 'net foward path' + '='*10)
    time1 = time.time()
    net.forward()
    fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
    scale_lst= [fuse]
    plot_single_scale(scale_lst, str(name_lst[idx]) + 'fuse')
    print('time elapsed: {}'.format(time.time()-time1))

    # print('python FASText/tools/test.py {}'.format(test_lst[idx]))
    os.system('python FASText/tools/test.py {}'.format(test_lst[idx]))
    out_ = np.array(Image.open('p2_out/{}.png'.format(name_lst[idx])))[:,:,::-1]
    cv2.imshow('output', out_)
    cv2.moveWindow('output', 500, 500)
    cv2.waitKey(vis_time)
cv2.destroyAllWindows()

# print('total time: {}'.format(time.time()-starttime))
