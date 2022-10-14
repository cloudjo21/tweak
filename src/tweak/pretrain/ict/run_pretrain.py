import argparse
import torch
import urllib.parse

from transformers import (
    AutoConfig,
    Trainer,
    set_seed,
)
from transformers.training_args import TrainingArguments

from tunip.service_config import get_service_config
from tunip.path.mart import MartPretrainedModelPath

from tweak.dataset.multitask_dataset import MultitaskResourceBuilder
from tweak.model.dpr.modeling_dpr import DPRForPreTraining
from tweak.utils.task_set_yaml_parser import TaskSetYamlParser
# from tweak.model.dpr.modeling_dpr import DPRContextEncoder
# from tweak.model.dpr.modeling_dpr import DPRQuestionEncoder

TARGET_TASK_NAME = 'ict'
ict_model_name = 'kodqa/dpr'


set_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

service_config = get_service_config()

parser = argparse.ArgumentParser()
parser.add_argument("--service_stage_config", type=str)
parser.add_argument("--task_config", type=str)
args = parser.parse_args()


# TODO enrole DPRForPreTraining to initialize question/context config
default_dpr_ctx_encoder_path = 'facebook/dpr-ctx_encoder-single-nq-base'
default_dpr_question_encoder_path = 'facebook/dpr-question_encoder-single-nq-base'

# load dpr config and save it
dpr_ctx_model_name = f'{ict_model_name}-ctx_encoder-single-nq-base'
dpr_ctx_encoder_path = f"{service_config.local_prefix}/{MartPretrainedModelPath(user_name=service_config.username, model_name=urllib.parse.quote(dpr_ctx_model_name, safe=''))}"

ctx_config = AutoConfig.from_pretrained(default_dpr_ctx_encoder_path)

ctx_config._name_or_path = dpr_ctx_model_name
ctx_config.save_pretrained(dpr_ctx_encoder_path)

dpr_question_model_name = f'{ict_model_name}-question_encoder-single-nq-base'
dpr_question_encoder_path = f"{service_config.local_prefix}/{MartPretrainedModelPath(user_name=service_config.username, model_name=urllib.parse.quote(dpr_question_model_name, safe=''))}"

question_config = AutoConfig.from_pretrained(default_dpr_question_encoder_path)

question_config._name_or_path = dpr_question_model_name
question_config.save_pretrained(dpr_question_encoder_path)
# END-OF-TODO


# load default models
# ctx_model = DPRContextEncoder(ctx_config)
# question_model = DPRQuestionEncoder(question_config)

model = DPRForPreTraining(ctx_config, question_config)

task_set_parser = TaskSetYamlParser(yaml_file=args.task_config, config=service_config.config)
task_set = task_set_parser.parse()

assert len(task_set.tasks) == 1
assert task_set.tasks[0] == TARGET_TASK_NAME
target_task_name = task_set.tasks[0].task_name

mt_resource_builder = MultitaskResourceBuilder(task_set)

trainer = Trainer(
    model=model.to(device),
    args=task_set.training_args,
    train_dataset=mt_resource_builder.train_dataset[target_task_name],
    eval_dataset=mt_resource_builder.validation_dataset[target_task_name],
    # compute_metrics=mt_resource_builder.compute_metrics
)
trainer.train()

trainer.save_model()
