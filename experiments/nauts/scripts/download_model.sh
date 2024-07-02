#!/bin/bash

export MODEL_NAME=monologg/koelectra-small-v3-discriminator

cd src && nohup python -m tweak.utils.save_model \
  --name=$MODEL_NAME \
  > ../download_model.nohup \
  2>&1 &
