#!/bin/bash
# CUDA_VISIBLE_DEVICES를 사용안하면 멀티 GPU를 사용하는 문제가 발생하여 해당 옵션 꼭 사용필요

CUDA_VISIBLE_DEVICES=1 python train.py \
  --model_name efficientnet_lite0 \
  --train_dir hymenoptera_data/train \
  --val_dir hymenoptera_data/val \
  --num_classes 2 \
  --batch_size 16 \
  --total_iters 10000 \
  --learning_rate 0.01 \
  --momentum 0.9 \
  --weight_decay 0.00004 \
  --label_smooth 0.1 \
  --num_workers 4 \
  --save ./models \
  --display_interval 20 \
  --val_interval 1000 \
  --save_interval 1000 \
  --auto_continue True \
