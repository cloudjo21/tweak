#!/bin/bash

export USERNAME=ed
export SERVICE_USERNAME=data
export BUCKET=

export MODEL_TYPE=hf.plm_model
export MODEL_NAME=cosmoquester/bart-ko-mini
export TOKENIZER_NAME=cosmoquester/bart-ko-mini
export MODEL_NAME_ESCAPED=cosmoquester%2Fbart-ko-mini
export MAX_LENGTH=128
export DEVICE=cuda
export ENCODER_ONLY=True


cd src && nohup python -m tweak.torch2onnx \
  --model_type=$MODEL_TYPE \
  --pt_model_name=$MODEL_NAME \
  --tokenizer_name=$TOKENIZER_NAME \
  --max_length=$MAX_LENGTH \
  --device=$DEVICE \
  --encoder_only=$ENCODER_ONLY \
  > ../torch2onnx.nohup \
  2>&1 &

tail -f torch2onnx.nohup


# Upload to GCS
echo "$NAUTS_LOCAL_ROOT/user/$USERNAME/mart/plm/models/$MODEL_NAME_ESCAPED/vocab gs://$BUCKET/user/$USERNAME/mart/plm/models/$MODEL_NAME_ESCAPED/vocab"
gsutil cp -r $NAUTS_LOCAL_ROOT/user/$USERNAME/mart/plm/models/$MODEL_NAME_ESCAPED/vocab gs://$BUCKET/user/$USERNAME/mart/plm/models/$MODEL_NAME_ESCAPED/vocab
gsutil cp -r $NAUTS_LOCAL_ROOT/user/$USERNAME/mart/plm/models/$MODEL_NAME_ESCAPED/config.json gs://$BUCKET/user/$USERNAME/mart/plm/models/$MODEL_NAME_ESCAPED/config.json
gsutil cp -r $NAUTS_LOCAL_ROOT/user/$USERNAME/mart/plm/models/$MODEL_NAME_ESCAPED/onnx gs://$BUCKET/user/$USERNAME/mart/plm/models/$MODEL_NAME_ESCAPED/onnx

# Transfer to Service Level Account
python -m inkling.utils.copy_files_gcs --source /user/${USERNAME}/mart/plm/models/${MODEL_NAME_ESCAPED} --target /user/${SERVICE_USERNAME}/mart/plm/models/${MODEL_NAME_ESCAPED}

# TODO YOU HAVE TO UPLOAD config.pbtxt of onnx model to GCS
