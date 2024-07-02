#!/bin/bash

export MODEL_NAME=facebook/rag-token-nq

cd src && nohup python -m tweak.utils.save_model \
  --name=$MODEL_NAME \
  > ../download_rag_model.nohup \
  2>&1 &
