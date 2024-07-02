"""
DEPRECATED
"""

import json
import logging
import torch
import urllib.parse
from pathlib import Path

from transformers import set_seed

from tunip.env import NAUTS_HOME
from tunip.logger import init_logging_handler
from tunip.model_uploader import ModelLoader
from tunip.path_utils import TaskPath
from tunip.path.mart import MartPretrainedModelPath
from tunip.snapshot_utils import SnapshotPathProvider

from tweak.utils.task_set_yaml_parser import TaskSetYamlParser
from tweak.dataset.multitask_dataset import MultitaskResourceBuilder
from tweak.model.multitask.modeling_multitask import MultitaskModel
from tweak.trainer.multitask_trainer import MultitaskTrainer
from tweak.data_collate import DataCollatorFactory, DummyDataCollator
from tweak.model.multitask.dump import MultitaskPredictionDumper

from tweak.model.convert.torchscript.service import TorchScriptModelConvertService
from tweak.model.convert.requests import Torch2TorchScriptRequest
from tweak.model.convert.torchscript.runtime_path import RuntimePathProviderFactory
from tweak.model.convert.torchscript import TorchScriptHfModelConverter


class Trainer:
    """
    TODO Deprecation
    """
    def __init__(self, service_config, task_config_path):
        self.service_config = service_config
        task_set_parser = TaskSetYamlParser(yaml_file=task_config_path, config=service_config)
        self.task_set = task_set_parser.parse()
        assert len(self.task_set.tasks) > 0

        
    def train(self):
        logger = init_logging_handler(name=f"{self.task_set.snapshot_dt}_train", level=logging.DEBUG)
        logger.info(self.task_set.training_args)

        set_seed(42)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        mt_resource_builder = MultitaskResourceBuilder(self.task_set)

        multitask_model = MultitaskModel.create(
            model_name=self.task_set.tasks[0].pretrained_model_name,
            model_types=mt_resource_builder.model_types,
            model_configs=mt_resource_builder.model_configs,
        )    

        data_collator_dict = {}
        for task_name in self.task_set.names:
            data_collator_dict[task_name] = DataCollatorFactory.create(
                # Ignoring task, use default data collator
                # TODO more test on DataCollatorForTokenClassification
                task_name=None,
                tokenizer=multitask_model.tokenizer,
            )

        trainer = MultitaskTrainer(
            task_name_list=self.task_set.names,
            mtl_data_collator=data_collator_dict,
            model=multitask_model.to(self.device),
            args=self.task_set.training_args,
            data_collator=DummyDataCollator(),
            train_dataset=mt_resource_builder.train_dataset,
            eval_dataset=mt_resource_builder.validation_dataset,
            compute_metrics=mt_resource_builder.compute_metrics
        )

        trainer.train()
        trainer.save_model()

        converter = TorchScriptModelConvertService(self.service_config)

        for task in self.task_set.tasks:
            # TODO: add various model types
            if task.task_type.name == "TOKEN_CLASSIFICATION":
                model_type = "hf.token_classification_model"

            conv_req = Torch2TorchScriptRequest(
                model_type=model_type,
                domain_name=self.task_set.domain_name,
                domain_snapshot=self.task_set.snapshot_dt,
                task_name=task.task_name,
                tokenizer_name=task.pretrained_model_name,
                pt_model_name=task.pretrained_model_name,
                max_length=task.max_length,
                checkpoint="",
                lang="ko"
            )
            converter(conv_req)

            model_path = str(TaskPath(self.service_config.username, self.task_set.domain_name, self.task_set.snapshot_dt, task.task_name))
            tokenizer_path = str(MartPretrainedModelPath(self.service_config.username, urllib.parse.quote(task.pretrained_model_name, safe='')))
            
            snapshot_path_provider = SnapshotPathProvider(self.service_config)
            lake_domain_path = f"/user/{self.service_config.username}/lake/document/{self.task_set.domain_name}/"
            
            predict_config_json = {
                "predict_config" : {
                    "predict_tokenizer_type": "auto",
                    "predict_model_type": "torchscript",
                    "predict_model_config": {
                        "model_path": model_path,
                        "task_name": task.task_name,
                        "task_type": task.task_type.name,
                    },
                    "tokenizer_config": {
                        "model_path": tokenizer_path,
                        "path": f"{tokenizer_path}/vocab",
                        "max_length": task.max_length
                    }
                },
                "domain_name": self.task_set.domain_name,
                "task_name": task.task_name,
                "batch_size": 20000
            }

            with open(f"{self.service_config.local_prefix}{model_path}/predict.json", "w") as pred_conf:
                json.dump(predict_config_json, pred_conf, indent=4)

        predict_results = trainer.predict(mt_resource_builder.test_dataset)

        eval_result_dumper = MultitaskPredictionDumper(self.service_config, self.task_set)
        eval_result_dumper.dump(
            mt_resource_builder.all_label_list,
            mt_resource_builder.test_dataset,
            predict_results
        )

        for task in self.task_set.tasks:
            model_uploader = ModelLoader(domain_name=self.task_set.domain_name,
                                         model_name=urllib.parse.quote(task.pretrained_model_name, safe=''),
                                         snapshot_dt=self.task_set.snapshot_dt,
                                         task_name=task.task_name)
            model_uploader.upload()

        return self.task_set