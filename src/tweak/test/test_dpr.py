import urllib.parse

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import AutoConfig
from transformers.models.dpr.configuration_dpr import DPRConfig

from tunip.service_config import get_service_config
from tunip.path.mart import MartPretrainedModelPath, MartTokenizerPath
from tunip.env import NAUTS_LOCAL_ROOT

service_config = get_service_config()
# model_name = 'hyunwoongko/kobart'
model_name = 'monologg/kobert'


plm_model_path = f"{service_config.local_prefix}/{MartPretrainedModelPath(user_name=service_config.username, model_name=urllib.parse.quote(model_name, safe=''))}"
tokenizer_path_or_pt_model_name = f"{service_config.local_prefix}/{MartTokenizerPath(user_name=service_config.username, tokenizer_name=model_name)}"


print(service_config.local_prefix)
print(str(plm_model_path))


# config = AutoConfig.from_pretrained(f"{plm_model_path}")
config = DPRConfig.from_pretrained(f"{plm_model_path}")
print(config)

tokenizer = DPRContextEncoderTokenizer.from_pretrained(config._name_or_path, config=config, use_fast=True)
model = DPRContextEncoder.from_pretrained(plm_model_path, config=config)

input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
embeddings = model(input_ids).pooler_output
print(embeddings.shape)

