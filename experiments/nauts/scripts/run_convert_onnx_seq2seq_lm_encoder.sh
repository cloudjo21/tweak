#!/bin/bash

export USERNAME=ed
export SERVICE_USERNAME=data
export BUCKET=

export MODEL_TYPE=hf.seq2seq_lm_model
export MODEL_NAME=hyunwoongko/kobart
export MODEL_NAME_ESCAPED=hyunwoongko%2Fkobart
export TOKENIZER_NAME=hyunwoongko/kobart

export DOMAIN_NAME=query2item_intro
export DOMAIN_SNAPSHOT=20230428_112957_673882

export TASK_NAME=generation
export MAX_LENGTH=128
export DEVICE=cuda
export ENCODER_ONLY=True


cd src && nohup python -m tweak.torch2onnx \
  --model_type=$MODEL_TYPE \
  --pt_model_name=$MODEL_NAME \
  --tokenizer_name=$TOKENIZER_NAME \
  --domain_name=$DOMAIN_NAME \
  --domain_snapshot=$DOMAIN_SNAPSHOT \
  --task_name=$TASK_NAME \
  --max_length=$MAX_LENGTH \
  --device=$DEVICE \
  --encoder_only=$ENCODER_ONLY \
  > ../torch2onnx.nohup \
  2>&1 &

tail -f torch2onnx.nohup


# Upload to GCS
echo "$NAUTS_LOCAL_ROOT/user/$USERNAME/mart/plm/models/$MODEL_NAME_ESCAPED/vocab gs://$BUCKET/user/$USERNAME/mart/plm/models/$MODEL_NAME_ESCAPED/vocab"
gsutil cp -r $NAUTS_LOCAL_ROOT/user/$USERNAME/mart/plm/models/$MODEL_NAME_ESCAPED/vocab gs://$BUCKET/user/$USERNAME/mart/plm/models/$MODEL_NAME_ESCAPED/vocab
gsutil cp -r $NAUTS_LOCAL_ROOT/user/$USERNAME/domains/$DOMAIN_NAME/$DOMAIN_SNAPSHOT/model/config.json gs://$BUCKET/user/$USERNAME/domains/$DOMAIN_NAME/$DOMAIN_SNAPSHOT/model/config.json
gsutil cp -r $NAUTS_LOCAL_ROOT/user/$USERNAME/domains/$DOMAIN_NAME/$DOMAIN_SNAPSHOT/model/$TASK_NAME/config.json gs://$BUCKET/user/$USERNAME/domains/$DOMAIN_NAME/$DOMAIN_SNAPSHOT/model/$TASK_NAME/config.json
gsutil cp -r $NAUTS_LOCAL_ROOT/user/$USERNAME/domains/$DOMAIN_NAME/$DOMAIN_SNAPSHOT/model/$TASK_NAME/onnx gs://$BUCKET/user/$USERNAME/domains/$DOMAIN_NAME/$DOMAIN_SNAPSHOT/model/$TASK_NAME/onnx

# Transfer to Service Level Account
python -m inkling.utils.copy_files_gcs --source /user/${USERNAME}/domains/${DOMAIN_NAME}/$DOMAIN_SNAPSHOT/model/config.json --target /user/${SERVICE_USERNAME}/domains/${DOMAIN_NAME}/model/
python -m inkling.utils.copy_files_gcs --source /user/${USERNAME}/domains/${DOMAIN_NAME}/$DOMAIN_SNAPSHOT/model/${TASK_NAME} --target /user/${SERVICE_USERNAME}/domains/${DOMAIN_NAME}/model/${TASK_NAME}

# TODO YOU HAVE TO UPLOAD config.pbtxt of onnx model to GCS
