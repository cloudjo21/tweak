#!/bin/bash

python -m tweak.model.multitask.train \
	--service_stage_config=/mnt/d/.nauts/experiments/ed/resources/application.json \
	--task_config=/mnt/d/.nauts/experiments/ed/resources/korean_sts/train.yml
