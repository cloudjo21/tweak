# tweak

## notes

- In macOS environment, you need to install Rust compiler as follows.

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Converting torch model to onnx model

### token classification model for onnx

```bash
#!/bin/bash

export MODEL_TYPE=hf.token_classification_model
export MODEL_NAME=klue/roberta-base
export TOKENIZER_NAME=klue/roberta-base
export DOMAIN_NAME=item_description
export DOMAIN_SNAPSHOT=20221116_205841_267193
export TASK_NAME=ner
export MAX_LENGTH=512
export DEVICE=cuda

cd src && nohup python -m tweak.torch2onnx \
  --model_type=$MODEL_TYPE \
  --pt_model_name=$MODEL_NAME \
  --tokenizer_name=$TOKENIZER_NAME \
  --domain_name=$DOMAIN_NAME \
  --domain_snapshot=$DOMAIN_SNAPSHOT \
  --task_name=$TASK_NAME \
  --max_length=$MAX_LENGTH \
  --device=$DEVICE \
  > torch2onnx.nohup \
  2>&1 &
```

## Converting torch model to torchscript model

### pretrained-language model

```bash
#!/bin/bash

export MODEL_NAME=snunlp/KR-ELECTRA-discriminator
export TOKENIZER_NAME=snunlp/KR-ELECTRA-discriminator

cd src && nohup python -m tweak.torch2torchscript_for_pretraining \
  --pt_model_name=$MODEL_NAME \
  --tokenizer_name=$TOKENIZER_NAME \
  > ../torch2torchscript_for_pretraining.nohup \
  2>&1 &
```

### token classification model for torchscript

```bash
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
  > torch2torchscript.nohup \
  2>&1 &
```
