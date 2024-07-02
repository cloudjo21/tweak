# TODO integrate requirements/variations of train*.py into only tweak/model/multitask/trainer.py

import argparse
import logging
import torch

from transformers import set_seed
from transformers import (
    Trainer
)

# from shellington.evaluate.multitask.evaluator import MultitaskEvaluator
# from shellington.evaluate.utils.eval_task_set_yaml_parser import EvalTaskSetYamlParser

from tunip.config import Config
from tunip.logger import init_logging_handler
from tunip.service_config import ServiceLevelConfig

from tweak.data_collate import DataCollatorFactory
from tweak.dataset.multitask_dataset import MultitaskResourceBuilder
from tweak.model.ffn import TargetValueModel
from tweak.model.multitask.dump import MultitaskPredictionDumper
from tweak.utils.task_set_yaml_parser import TaskSetYamlParser


set_seed(42)


parser = argparse.ArgumentParser()
parser.add_argument("--service_stage_config", type=str)
parser.add_argument("--model_config", type=str)
parser.add_argument("--task_config", type=str)
parser.add_argument("--eval_task_config", type=str, default='')
args = parser.parse_args()


# getting configuratinos from service stage configuration file
# application*.json
service_config = ServiceLevelConfig(config=Config(args.service_stage_config))

# TRAINING TASK
# getting configuratinos from task configuration file
# [domain_name]_[task_name]_train_task.yaml
task_set_parser = TaskSetYamlParser(yaml_file=args.task_config, config=service_config.config)
task_set = task_set_parser.parse()
assert len(task_set.tasks) == 1

task_name = task_set.tasks[0].task_name


# init logger with snapshot_dt
logger = init_logging_handler(name=f"{task_set.snapshot_dt}_train", level=logging.DEBUG)
logger.info(task_set.training_args)

model_config = Config(args.model_config)
target_model = TargetValueModel(config=model_config)

# supports to build some resources like datasets, metrics, and configs
mt_resource_builder = MultitaskResourceBuilder(task_set)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

trainer = Trainer(
    model=target_model.to(device),
    args=task_set.training_args,
    data_collator=DataCollatorFactory.create(task_name),
    train_dataset=mt_resource_builder.train_dataset[task_name],
    eval_dataset=mt_resource_builder.validation_dataset[task_name],
    compute_metrics=mt_resource_builder.compute_metrics[task_name],
)

# TODO support resume_from_checkpoint
trainer.train()
trainer.save_model()

predict_results = trainer.predict(mt_resource_builder.test_dataset[task_name])


# Dump evaluation results
eval_result_dumper = MultitaskPredictionDumper(service_config, task_set)
eval_result_dumper.dump(
    mt_resource_builder.all_label_list,
    mt_resource_builder.test_dataset,
    {task_name: predict_results}
)
