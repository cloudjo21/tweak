#!/bin/bash

export MODEL_NAME=snunlp/KR-ELECTRA-discriminator

cd src && nohup python -m tweak.utils.save_model \
  --name=$MODEL_NAME \
  > ../download_model.nohup \
  2>&1 &
