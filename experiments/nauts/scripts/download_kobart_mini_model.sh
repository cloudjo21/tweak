#!/bin/bash

export MODEL_NAME=cosmoquester/bart-ko-mini

cd src && nohup python -m tweak.utils.save_model \
  --name=$MODEL_NAME \
  > ../download_model.nohup \
  2>&1 &
