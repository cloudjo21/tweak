#!/bin/bash

export MODEL_TYPE=sklearn.classification_xgb
export DOMAIN_NAME=item-reco
export DOMAIN_SNAPSHOT=20230626_145600_000000
export TASK_NAME=classification

export TIMESTAMP=$(date +%s)

cd src && nohup python -m tweak.sklearn2onnx \
  --model_type=$MODEL_TYPE \
  --domain_name=$DOMAIN_NAME \
  --domain_snapshot=$DOMAIN_SNAPSHOT \
  --task_name=$TASK_NAME \
  > sklearn2onnx.$TIMESTAMP.nohup \
  2>&1 &
