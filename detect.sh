#!/usr/bin/env bash
DATA="./ori"
IMG=$1
DST1="./p1_out"
DST2="./p2_out"
#WRN='wrn.txt'

CUDA_VISIBLE_DEVICES=0 python src/demo.py \
--img_name ${IMG}.png \
--data_root $DATA \
--dest_folder $DST1
#2>${WRN}

feh ${DST1}/${IMG}*.png

CUDA_VISIBLE_DEVICES=0 python FASText/tools/test.py ${DATA} ${DST1} ${DST2}

feh ${DST2}/${IMG}*.png
