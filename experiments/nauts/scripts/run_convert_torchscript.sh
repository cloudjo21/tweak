#!/bin/bash

export MODEL_TYPE=hf.token_classification_model
export MODEL_NAME=klue/roberta-base
export TOKENIZER_NAME=klue/roberta-base
export DOMAIN_NAME=item_description
export DOMAIN_SNAPSHOT=20221116_205841_267193
export TASK_NAME=ner
export MAX_LENGTH=512
export DEVICE=cuda

cd src && nohup python -m tweak.torch2torchscript \
  --model_type=$MODEL_TYPE \
  --pt_model_name=$MODEL_NAME \
  --tokenizer_name=$TOKENIZER_NAME \
  --domain_name=$DOMAIN_NAME \
  --domain_snapshot=$DOMAIN_SNAPSHOT \
  --task_name=$TASK_NAME \
  --max_length=$MAX_LENGTH \
  --device=$DEVICE \
  > ../torch2torchscript.nohup \
  2>&1 &
