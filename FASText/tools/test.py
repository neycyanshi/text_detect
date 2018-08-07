'''
Created on Jan 8, 2015

@author: busta
'''
import numpy as np
import cv2
import sys
import os
from ft import FASTex
from vis import draw_keypoints
import matplotlib.pyplot as plt
import time
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = False


def find_rect(cnt,T):
    rect = cv2.minAreaRect(cnt)
    rect_area = rect[1][0]*rect[1][1]
    cnt_area = cv2.contourArea(cnt)
    if rect_area*T > cnt_area:
        return cnt
    box = np.int0(cv2.boxPoints(rect))
    return box

def draw_contours(imgFile,srcFile,dstFile,pic):
    img = cv2.imread(srcFile+'/'+pic)
    imgc = cv2.imread(imgFile+'/'+pic[:-9]+'.png')
    imgd = dstFile + '/' + pic[:-9]+'.png'
    

    #print(out)
    segmentations = ft.getCharSegmentations(cv2.cvtColor(imgc,cv2.COLOR_BGR2GRAY), outputDir, 'base')
    keypoints = ft.getLastDetectionKeypoints()   
    octave0Points = keypoints[keypoints[:,2] == 0,:]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,binary = cv2.threshold(gray,gray_threshold,255,cv2.THRESH_BINARY) 
    #print binary.shape,np.max(binary)
    _,contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key = lambda x:cv2.contourArea(x),reverse = True)
    keypoints = [[int(s[0]),int(s[1]),-1] for s in octave0Points]
    #print kp
    contours = [cnt for cnt in contours if len(cnt) > 10]
    if len(contours) == 0:
        cv2.imwrite(imgd,imgc)
        cv2.imwrite(dstFile+'/'+pic[:-9]+'fuse0.png',img)
        return
    cnts = zip(contours,range(len(contours)))
    kp_list = []
    for kp in keypoints:
        for cnt in cnts:
            if cv2.pointPolygonTest(cnt[0],(kp[0],kp[1]),False) >= 0:  
                kp[2] = cnt[1]
                kp_list.append(kp[2])
                break
    kp_list = set(kp_list)
    cnt_list = [cnt for cnt in cnts if cnt[1] in kp_list or cv2.contourArea(cnt[0]) > 180]
    #print 'pic',pic
    #for cnt in cnt_list:
    #    print cv2.contourArea(cnt[0])
    if len(cnt_list) != 0:
        contours,k = zip(*cnt_list) 
    contours = [find_rect(cnt,T) for cnt in contours]
    mask = np.zeros(gray.shape,np.uint8)
    cv2.drawContours(mask,contours,-1,255,-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    dilate = cv2.dilate(mask,kernel)
    _,contours,_ = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgc,contours,-1,(0,0,255),2)
    cv2.imwrite(imgd,imgc)
    
    ss = np.zeros(gray.shape,np.uint8)
    cv2.drawContours(ss,contours,-1,255,-1)
    cv2.imwrite(dstFile+'/'+pic[:-9]+'mask.png',ss)
    
    # cv2.imwrite(dstFile+'/'+pic[:-9]+'fuse0.png',img)


if __name__ == '__main__':
    
    outputDir = './tmp'
    edgeThreshold = 13
    T = 0.8
    gray_threshold = 180

    ft = FASTex(edgeThreshold= edgeThreshold, nlevels=-1, minCompSize = 4)
    
    imgName = '/datagrid/personal/TextSpotter/evaluation-sets/bornDigital/img_100.png'
        
    if len(sys.argv) == 1:
        if sys.argv[1].endswith(".png") or sys.argv[1].endswith(".jpg"):
            imgName = sys.argv[1]
    if len(sys.argv) >= 2:
            imgFile = sys.argv[1]
            srcFile = sys.argv[2]
            dstFile = sys.argv[3]
    print imgFile,srcFile,dstFile
    if not os.path.isdir(dstFile):
        os.makedirs(dstFile)
    starttime = time.time()
    n_im = 0
    for pic in os.listdir(srcFile):
        if pic.endswith("fuse0.png"):
            n_im += 1
            draw_contours(imgFile,srcFile,dstFile,pic)
    endtime = time.time()
    total_time = endtime - starttime
    print('='*10)
    print 'stroke detect total time: {} for {} imgs.'.format(total_time, n_im)
    print('stroke detect avg_time: {}'.format(total_time / n_im))
    print('='*10)

    #draw_keypoints(imgc, octavePoints, edgeThreshold, inter = True, color = 0)
