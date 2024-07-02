#!/bin/bash

export MODEL_NAME=hyunwoongko/kobart
export TOKENIZER_NAME=hyunwoongko/kobart


cd src && nohup python -m tweak.torch2torchscript_for_pretraining \
  --pt_model_name=$MODEL_NAME \
  --tokenizer_name=$TOKENIZER_NAME \
  > ../torch2torchscript_for_pretraining.nohup \
  2>&1 &
