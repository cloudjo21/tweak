import torch
import urllib.parse
from transformers import RagConfig, RagTokenizer, RagTokenForGeneration, RagRetriever

from tunip.path.mart import MartPretrainedModelPath, MartTokenizerPath
from tunip.service_config import get_service_config


model_name = 'facebook/rag-token-nq'
index_name = 'exact'
use_dummy_dataset = True

service_config = get_service_config()
model_name = urllib.parse.quote(model_name, safe='')

plm_model_path = f"{service_config.local_prefix}/{str(MartPretrainedModelPath(user_name=service_config.username, model_name=model_name))}"

tokenizer_path_or_pt_model_name = f"{service_config.local_prefix}/{MartTokenizerPath(user_name=service_config.username, tokenizer_name=model_name)}"

config = RagConfig.from_pretrained(plm_model_path)
# tokenizer = RagTokenizer.from_pretrained(tokenizer_path_or_pt_model_name, config=config)
# qe_tokenizer = RagTokenizer.from_pretrained(f"{plm_model_path}/vocab/question_encoder_tokenizer", config=config)
# gn_tokenizer = RagTokenizer.from_pretrained(f"{plm_model_path}/vocab/generator_tokenizer", config=config)
# qe_tokenizer = RagTokenizer.from_pretrained(f"{plm_model_path}/vocab", config=config)
tokenizer = RagTokenizer.from_pretrained(f"{plm_model_path}/vocab", config=config, use_fast=True)
qe_tokenizer = tokenizer.question_encoder
gn_tokenizer = tokenizer.generator
retriever = RagRetriever.from_pretrained(
    f"{plm_model_path}/retrieve", config=config,
    index_name=index_name, use_dummy_dataset=use_dummy_dataset
)
model = RagTokenForGeneration.from_pretrained(plm_model_path, config=config)

inputs = qe_tokenizer("How many people live in Paris?", return_tensors="pt")
input_ids = inputs["input_ids"]

# targets = gn_tokenizer("In Paris, there are 10 million people.", return_tensors="pt")
# labels = targets["input_ids"]
# outputs = model(input_ids=input_ids, labels=labels)

# 1. Encode
question_hidden_states = model.question_encoder(input_ids)[0]
# 2. Retrieve
docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
doc_scores = torch.bmm(
    question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
).squeeze(1)

# or directly generate
generated = model.generate(
    context_input_ids=docs_dict["context_input_ids"],
    context_attention_mask=docs_dict["context_attention_mask"],
    doc_scores=doc_scores,
)
generated_string = gn_tokenizer.batch_decode(generated, skip_special_tokens=True)

print(generated_string)
