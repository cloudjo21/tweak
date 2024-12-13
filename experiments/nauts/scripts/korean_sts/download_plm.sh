#!/bin/bash

export MODEL_NAME=jhgan/ko-sroberta-multitask

nohup python -m tweak.utils.save_model \
  --name=$MODEL_NAME \
  > save_model_plm.nohup \
  2>&1 &
