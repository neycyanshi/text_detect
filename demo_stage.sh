#!/usr/bin/env bash
DATA="ori"
DST1="p1_out"
DST2="p2_out"

CUDA_VISIBLE_DEVICES=0 python src/demo_stage.py \
--data_root ${DATA} \
--dest_folder ${DST1}
