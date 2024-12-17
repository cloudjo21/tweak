import argparse
import logging
import torch

from transformers import set_seed

# from shellington.evaluate.multitask.evaluator import MultitaskEvaluator
# from shellington.evaluate.utils.eval_task_set_yaml_parser import EvalTaskSetYamlParser

from tunip.config import Config
from tunip.logger import init_logging_handler
from tunip.service_config import ServiceLevelConfig

from tweak.data_collate import DataCollatorFactory, DummyDataCollator
from tweak.dataset.multitask_dataset import MultitaskResourceBuilder
from tweak.model.multitask.dump import MultitaskPredictionDumper
from tweak.model.multitask.modeling_multitask import MultitaskModel
from tweak.trainer.multitask_trainer import MultitaskTrainer
from tweak.utils.task_set_yaml_parser import TaskSetYamlParser


set_seed(42)


parser = argparse.ArgumentParser()
parser.add_argument("--service_stage_config", type=str)
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
assert len(task_set.tasks) > 0

# init logger with snapshot_dt
logger = init_logging_handler(name=f"{task_set.snapshot_dt}_train", level=logging.DEBUG)
logger.info(task_set.training_args)

# import transformers
# transformers.utils.logging.set_verbosity('debug')

# supports to build some resources like datasets, metrics, and configs
mt_resource_builder = MultitaskResourceBuilder(task_set)

# create the model for multi-task learning
# TODO support the init. by MultitaskResourceBuilder
multitask_model = MultitaskModel.create(
    model_name=task_set.tasks[0].pretrained_model_name,
    model_types=mt_resource_builder.model_types,
    model_configs=mt_resource_builder.model_configs,
)

# data collator from datasets
# TODO support the init. by MultitaskResourceBuilder
data_collator_dict = {}
for task_name in task_set.names:
    data_collator_dict[task_name] = DataCollatorFactory.create(
        # Ignoring task, use default data collator
        # TODO more test on DataCollatorForTokenClassification
        task_name=None,
        tokenizer=multitask_model.tokenizer,
    )

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

trainer = MultitaskTrainer(
    task_name_list=task_set.names,
    mtl_data_collator=data_collator_dict,
    model=multitask_model.to(device),
    args=task_set.training_args,
    data_collator=DummyDataCollator(),
    train_dataset=mt_resource_builder.train_dataset,
    eval_dataset=mt_resource_builder.validation_dataset,
    compute_metrics=mt_resource_builder.compute_metrics,
    resume_from_checkpoint=task_set.resume_from_checkpoint
)

trainer.train()
# trainer.save_model()
# since transformers==4.7.x ?
# trainer.save_state()
# TODO multitask_model.tokenizer.save_vocabulary(task_set.training_args.output_dir)

predict_results = trainer.predict(mt_resource_builder.test_dataset)


# Dump evaluation results
eval_result_dumper = MultitaskPredictionDumper(service_config, task_set)
eval_result_dumper.dump(
    mt_resource_builder.all_label_list,
    mt_resource_builder.test_dataset,
    predict_results
)


# # TODO migrate to shellington
# # EVALUATION TASK
# # [domain_name]_[task_name]_eval_task.yaml
# if args.eval_task_config:
#     eval_task_set_parser = EvalTaskSetYamlParser(yaml_file=args.eval_task_config, config=config)
#     eval_task_set = eval_task_set_parser.parse()
#     assert len(eval_task_set.tasks) > 0

#     # set the snapshot_dt of trained model to evaluate
#     eval_task_set.snapshot_dt = task_set.snapshot_dt

#     # Evaluator
#     evaluator = MultitaskEvaluator(config, eval_task_set)
#     evaluator.evaluate()
