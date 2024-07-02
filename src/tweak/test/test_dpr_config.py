import transformers
import urllib.parse


dpr_ctx_model_name = 'facebook/dpr-ctx_encoder-single-nq-base'
urllib.parse.quote(dpr_ctx_model_name, safe='')
dpr_ctx_encoder_path = ''
ctx_config = transformers.AutoConfig.from_pretrained(dpr_ctx_encoder_path)
ctx_config.save_pretrained()

dpr_question_model_name = 'facebook/dpr-question_encoder-single-nq-base'
urllib.parse.quote(dpr_question_model_name, safe='')
dpr_question_encoder_path = ''
question_config = transformers.AutoConfig.from_pretrained(dpr_question_encoder_path)
question_config.save_pretrained()

