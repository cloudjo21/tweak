#!/bin/bash

export MODEL_TYPE=hf.token_classification_model
export MODEL_NAME=monologg/koelectra-small-v3-discriminator
export TOKENIZER_NAME=monologg/koelectra-small-v3-discriminator
export DOMAIN_NAME=item_description
export DOMAIN_SNAPSHOT=20221223_143738_452012
export TASK_NAME=ner
export MAX_LENGTH=128
export DEVICE=cuda

cd src && nohup python -m tweak.torch2onnx \
  --model_type=$MODEL_TYPE \
  --pt_model_name=$MODEL_NAME \
  --tokenizer_name=$TOKENIZER_NAME \
  --domain_name=$DOMAIN_NAME \
  --domain_snapshot=$DOMAIN_SNAPSHOT \
  --task_name=$TASK_NAME \
  --max_length=$MAX_LENGTH \
  --device=$DEVICE \
  > torch2onnx.nohup \
  2>&1 &
