# TODO move to utils
import argparse
import json
import os
import urllib.parse

from transformers import AutoModel, AutoTokenizer
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

from tunip.path.mart import MartPretrainedModelPath, MartTokenizerPath
from tunip.service_config import get_service_config

from tweak import LOGGER


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
        try:
            tokenizer.save_vocabulary(save_directory=tokenizer_path_or_pt_model_name)
        except NotImplementedError as nie:
            LOGGER.warning(str(nie))
            LOGGER.warning(f"{tokenizer.__class__}:save_vocabulary is not implemented yet")
            with open(f"{plm_model_path}/vocab/tokenizer.json") as tokenizer_f:
                tokenizer_json = json.load(tokenizer_f)
                vocabs = list(tokenizer_json['model']['vocab'].keys())
                with open(f"{tokenizer_path_or_pt_model_name}/vocab.txt", mode='w') as vocab_f:
                    vocab_f.write('\n'.join(vocabs))
        LOGGER.info(f"It is done to donwload model_name: {self.model_name} at {plm_model_path}!")
        LOGGER.info(f"It is done to donwload tokenizer at {tokenizer_path_or_pt_model_name}!")


class RagModelDownloader:
    def __init__(self, service_config, job_config):
        self.service_config = service_config
        self.model_name = job_config['pretrained_model_name']
        self.tokenizer_path = job_config.get('tokenizer_path')
    
    def download(self):

        model_name = urllib.parse.quote(self.model_name, safe='')
        plm_model_path = f"{self.service_config.local_prefix}/{str(MartPretrainedModelPath(user_name=self.service_config.username, model_name=model_name))}"

        tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        if not self.tokenizer_path:
            model_name_encoded = urllib.parse.quote(self.model_name, safe='')
            tokenizer_path_or_pt_model_name = f"{self.service_config.local_prefix}/{MartTokenizerPath(user_name=self.service_config.username, tokenizer_name=model_name_encoded)}"

        os.makedirs(tokenizer_path_or_pt_model_name, exist_ok=True)

        retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq",
            dataset='wiki_dpr',
            index_name="compressed"
            # index_name="exact", use_dummy_dataset=True
        )
        
        # or use retriever separately
        model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)

        model.save_pretrained(plm_model_path)
        tokenizer.save_pretrained(f"{plm_model_path}/vocab")
        LOGGER.info(f"type-of-tokenizer: {type(tokenizer)}")
        try:
            tokenizer.save_vocabulary(save_directory=tokenizer_path_or_pt_model_name)
        except NotImplementedError as nie:
            LOGGER.warning(str(nie))
            LOGGER.warning(f"{tokenizer.__class__}:save_vocabulary is not implemented yet")
            
        retriever.save_pretrained(f"{plm_model_path}/retrieve")
        # TODO support to save index of retriever?
        # retriever.index

        LOGGER.info(f"It is done to donwload model_name: {self.model_name} at {plm_model_path}!")
        LOGGER.info(f"It is done to donwload tokenizer at {tokenizer_path_or_pt_model_name}!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="plm model and vocabulary downloader")
    parser.add_argument(
        "-n",
        "--name",
        help="the name of pretraining language model",
        type=str,
        required=True
    )

    args = parser.parse_args()

    if 'rag' in args.name.lower():
        plm_model = RagModelDownloader(get_service_config(), {'pretrained_model_name': args.name})
        plm_model.download()
    else:
        plm_model = HfPretrainedModelDownloader(get_service_config(), {'pretrained_model_name': args.name})
        plm_model.download()
