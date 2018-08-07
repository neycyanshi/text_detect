from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io
import os
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_name', type=str, default='10.png', help='test img name')
parser.add_argument('--data_root', type=str, default='ori', help='path to data root folder.')
parser.add_argument('--dest_folder', type=str, default='p1_out', help='path to save side-output and fused result.')
parser.add_argument('--model_root', type=str, default='model', help='root for deploy.txt and caffemodel.')
parser.add_argument('--caffe_root', type=str, default='~/caffe', help='root path for your caffe.')

args = parser.parse_args()
print(args)

if not os.path.exists(args.dest_folder):
    os.makedirs(args.dest_folder)
    print('making new dest_folder: {}'.format(args.dest_folder))

caffe_root = args.caffe_root
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

def plot_single_scale(scale_lst, size, name):
    pylab.rcParams['figure.figsize'] = size, size/2
    for i in range(0, len(scale_lst)):
        image = Image.fromarray(255*(scale_lst[i])).convert('L')
        img_path = os.path.join(args.dest_folder, str(name) + str(i) + '.png')
        print('img saving to: {}'.format(img_path))
        image.save(img_path,'png')

data_root = args.data_root
test_lst = [os.path.join(data_root, args.img_name)]
name_lst = [os.path.basename(x).split('.')[0] for x in test_lst]
print('test_lst: {}'.format(test_lst))
print('name_lst: {}'.format(name_lst))

im_lst = []
print 'im_lst len' +  str(len(test_lst))
for i in range(0, len(test_lst)):
    print(test_lst[i])
    im = Image.open(test_lst[i])
    #if sz[0] > sz[1]:
    #        im = im.resize((960,int(sz[1]*960/sz[0])),Image.ANTIALIAS)
    #    else:
    #        im = im.resize((int(sz[0]*960/sz[1]),960),Image.ANTIALIAS)
    sz = im.size
#    if sz[0]*sz[1] > 2500000:
#        im = im.resize((int(sz[0]/2),int(sz[1]/2)),Image.ANTIALIAS)
    in_ = np.array(im,dtype=np.float32)
    print 'shape',in_.shape
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    im_lst.append(in_)

caffe.set_mode_gpu()
caffe.set_device(0)

model_root = args.model_root
net_path = os.path.join(model_root, 'deploy.prototxt')
# print(os.path.realpath(net_path))
weight_path = os.path.join(model_root, 'hed_pretrained_bsds.caffemodel')
#print('='*10 + net_path)
#print('='*10 + weight_path)
net = caffe.Net(net_path, weight_path, caffe.TEST)

starttime = datetime.datetime.now()
for idx in range(0,len(im_lst)):
    print idx
    in_ = im_lst[idx]
    in_ = in_.transpose((2,0,1))
    net.blobs['data'].reshape(1,*in_.shape)
    net.blobs['data'].data[...] = in_

    print('='*10 + 'net foward path' + '='*10)
    net.forward()
    out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
    out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
    out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
    out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
    out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]

    fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
    scale_lst= [fuse]
    plot_single_scale(scale_lst, 22, str(name_lst[idx]) + 'fuse')
    scale_lst = [out1,out2,out3,out4,out5]
    plot_single_scale(scale_lst, 10, str(name_lst[idx]) + 'each')
endtime = datetime.datetime.now()
print 'time: '+ str((endtime-starttime).seconds),endtime-starttime
