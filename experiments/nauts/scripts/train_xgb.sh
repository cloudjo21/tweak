#!/bin/bash

NOW=$(date +%Y%m%d-%H%M%S)

USERNAME=ed
DOMAIN_NAME=item-reco
TASK_NAME=regression

nohup python -m tweak.model.transfer.train_xgb \
        --task_config=/data/home/$USERNAME/.nauts/experiments/$USERNAME/resources/$DOMAIN_NAME/$TASK_NAME/train_xgb.yml \
        --service_stage_config=/data/home/$USERNAME/.nauts/experiments/$USERNAME/resources/application.json \
        > nohup.${DOMAIN_NAME}.{$TASK_NAME}.${NOW}out &

tail -f $NOW
