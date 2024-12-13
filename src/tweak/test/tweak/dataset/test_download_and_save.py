# require ir_datasets package
# pip install ir_datasets

import os

from datasets import load_dataset

from tunip.env import NAUTS_LOCAL_ROOT
from tunip.service_config import get_service_config


dataset = load_dataset('irds/msmarco-passage')

os.mkdir(f'{NAUTS_LOCAL_ROOT}/user/{get_service_config().username}/mart/msmarco')
dataset.save_to_disk(f'{NAUTS_LOCAL_ROOT}/user/{get_service_config().username}/mart/msmarco')

