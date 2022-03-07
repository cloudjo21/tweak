# TODO move to utils
import os
import urllib.parse

from transformers import AutoModel, AutoTokenizer

from tunip.path.mart import MartPretrainedModelPath, MartTokenizerPath
from tunip.service_config import get_service_config
from tunip.snapshot_utils import snapshot_now


class HfPretrainedModelDownloader:
    def __init__(self, service_config, job_config):
        self.service_config = service_config
        self.model_name = job_config['pretrained_model_name']
        self.tokenizer_path = job_config.get('tokenizer_path')

    def download(self):
        plm_model = AutoModel.from_pretrained(self.model_name)
        model_name = urllib.parse.quote(self.model_name, safe='')
        plm_model_path = f"{self.service_config.local_prefix}/{str(MartPretrainedModelPath(user_name=self.service_config.username, model_name=model_name))}"

        if not self.tokenizer_path:
            model_name_encoded = urllib.parse.quote(self.model_name, safe='')
            tokenizer_path_or_pt_model_name = f"{self.service_config.local_prefix}/{MartTokenizerPath(user_name=self.service_config.username, tokenizer_name=model_name_encoded)}"

        os.makedirs(tokenizer_path_or_pt_model_name, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        plm_model.save_pretrained(plm_model_path)
        tokenizer.save_pretrained(f"{plm_model_path}/vocab")
        tokenizer.save_vocabulary(save_directory=tokenizer_path_or_pt_model_name)

# plm_model = HfPretrainedModelDownloader(get_service_config(), {'pretrained_model_name': 'monologg/koelectra-small-v3-discriminator'})
# plm_model.download()
