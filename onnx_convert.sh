#!/bin/bash

# 변수 설정
MODEL_NAME="efficientnet_lite0"
CHECKPOINT_PATH="./models/efficientnet_lite0_best.pt"
SAVE_PATH="./models/efficientnet_lite0_best.onnx"

# Python 스크립트 실행
python onnx_convert.py \
  --model_name ${MODEL_NAME} \
  --checkpoint_path ${CHECKPOINT_PATH} \
  --save_path ${SAVE_PATH}