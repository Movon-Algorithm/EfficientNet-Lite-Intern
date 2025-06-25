#!/bin/bash

# ONNX 모델 파일 경로 지정
MODEL_PATH="./models/efficientnet_lite0_best.onnx"

# 입력 이미지 크기 지정 (기본 224)
INPUT_SIZE=224

# Python 추론 스크립트 실행
python onnx_inference_timecheck.py \
    --model_path ${MODEL_PATH} \
    --input_size ${INPUT_SIZE}
