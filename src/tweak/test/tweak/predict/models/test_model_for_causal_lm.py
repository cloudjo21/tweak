import unittest

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from tunip.env import NAUTS_LOCAL_ROOT
from tunip.service_config import get_service_config
from tweak.predict.models import PreTrainedModelConfig
from tweak.predict.resource_materialize import ResourceMaterializer


class ModelForCausalLMTest(unittest.TestCase):

    def setUp(self):
        self.service_config = get_service_config(force_service_level='dev')
        self.plm_model_path = f"/user/{self.service_config.username}/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator"
        config_json = {
            "model_path": self.plm_model_path,
            "model_name": "monologg/koelectra-small-v3-discriminator"
        }

        config = PreTrainedModelConfig.model_validate(config_json)
        ResourceMaterializer.apply_for_hf_model(config, self.service_config)

    def test_create_config(self):
        config = AutoConfig.from_pretrained(f"{NAUTS_LOCAL_ROOT}/{self.plm_model_path}")
        assert config

    def test_create_tokenizer(self):
        config = AutoConfig.from_pretrained(f"{NAUTS_LOCAL_ROOT}/{self.plm_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=f"{NAUTS_LOCAL_ROOT}/{self.plm_model_path}/vocab", config=config)
        assert tokenizer

    def test_create_model(self):
        # config_path = vocab_path = model_path = 'sberbank-ai/mGPT'
        config_path = vocab_path = model_path = 'skt/kogpt2-base-v2'

        config = AutoConfig.from_pretrained(config_path)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=vocab_path, config=config,
            bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>'
        )
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path, config=config)

        inputs = tokenizer('근육이 커지기 위해서는', return_tensors='pt')
        outputs = model(**inputs)

        id2token = dict(map(reversed, tokenizer.vocab.items()))
        output_tokens = [id2token[id] for id in outputs.logits[0].argmax(axis=-1).tolist()]
        print(output_tokens)
        assert model

    def test_decode_gpt2(self):
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token='</s>', eos_token='</s>', unk_token='<unk>',
            pad_token='<pad>', mask_token='<mask>')
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        text = '근육이 커지기 위해서는'
        input_ids = tokenizer.encode(text, return_tensors='pt')
        gen_ids = model.generate(input_ids,
            max_length=128,
            repetition_penalty=2.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            use_cache=True) # do_sample=False
        generated = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
        # generated = tokenizer.decode(gen_ids[0]) # skip_special_tokens=True
        print(generated)

    def test_predict_model(self):
        pass
