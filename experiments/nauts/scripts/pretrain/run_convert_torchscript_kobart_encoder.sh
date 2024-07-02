#!/bin/bash

export MODEL_NAME=hyunwoongko/kobart
export TOKENIZER_NAME=hyunwoongko/kobart
export ENCODER_ONLY=True


cd src && nohup python -m tweak.torch2torchscript_for_pretraining \
  --pt_model_name=$MODEL_NAME \
  --tokenizer_name=$TOKENIZER_NAME \
  --encoder_only=$ENCODER_ONLY \
  > ../torch2torchscript_for_pretraining.nohup \
  2>&1 &
