import os

from datasets import load_dataset

from tunip.env import NAUTS_LOCAL_ROOT
from tunip.service_config import get_service_config


#dataset = load_dataset('kor_nli', 'snli')
#os.mkdir(f'{NAUTS_LOCAL_ROOT}/user/{get_service_config().username}/mart/kor_nli_snli')
#dataset.save_to_disk(f'{NAUTS_LOCAL_ROOT}/user/{get_service_config().username}/mart/kor_nli_snli')

dataset = load_dataset('kor_nli', 'xnli')
#os.mkdir(f'{NAUTS_LOCAL_ROOT}/user/{get_service_config().username}/mart/kor_nli_xnli')
dataset.save_to_disk(f'{NAUTS_LOCAL_ROOT}/user/{get_service_config().username}/mart/kor_nli_xnli')

