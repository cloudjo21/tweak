#!/bin/bash

export MODEL_TYPE=hf.seq2seq_lm_model
export MODEL_NAME=hyunwoongko/kobart
export CHECKPOINT=checkpoint-10000
export TOKENIZER_NAME=hyunwoongko/kobart
export DOMAIN_NAME=query2item_intro
export DOMAIN_SNAPSHOT=20230328_130916_874499
export TASK_NAME=generation
export MAX_LENGTH=128
export DEVICE=cuda
export ENCODER_ONLY=True

cd src && nohup python -m tweak.torch2torchscript \
  --model_type=$MODEL_TYPE \
  --pt_model_name=$MODEL_NAME \
  --checkpoint=$CHECKPOINT \
  --tokenizer_name=$TOKENIZER_NAME \
  --domain_name=$DOMAIN_NAME \
  --domain_snapshot=$DOMAIN_SNAPSHOT \
  --task_name=$TASK_NAME \
  --max_length=$MAX_LENGTH \
  --device=$DEVICE \
  --encoder_only=$ENCODER_ONLY \
  > ../torch2torchscript.nohup \
  2>&1 &
