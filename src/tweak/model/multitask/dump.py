from pathlib import Path

import tunip.file_utils as file_utils

from tunip.config import Config
from tunip import ServiceLevelConfig
from tunip.file_utils import LocalFileHandler
from tunip.path_utils import NautsPathFactory
from tweak.task.task_set import TaskSet


class MultitaskPredictionDumper:
    def __init__(self, config: ServiceLevelConfig, task_set: TaskSet):
        self.hdfs_username = config.username
        self.file_loader: LocalFileHandler  = file_utils.services.get("LOCAL", config=config.config)
        assert self.file_loader.__class__ == LocalFileHandler

        self.task_set = task_set

    def dump(self, label_list_dict, test_dataset_dict, predict_result_dict):
        # TODO implement and use task-specific prediction dumper for each task
        for task in self.task_set:
            task_name = task.task_name
            task_path = NautsPathFactory.create_training_family(
                user_name=self.hdfs_username,
                domain_name=self.task_set.domain_name,
                snapshot_dt=self.task_set.snapshot_dt,
                task_name=task_name
            )
            output_path = Path(repr(task_path))

            self.file_loader.mkdirs(output_path)
            self.file_loader.save_pickle(str(output_path / "label_list.pickle"), label_list_dict[task_name])

            # Dump predictions, labels, test_dataset
            predictions = predict_result_dict[task_name].predictions
            label_ids = predict_result_dict[task_name].label_ids
            metrics = predict_result_dict[task_name].metrics

            self.file_loader.save_pickle(str(output_path / "predictions.pickle"), predictions)
            self.file_loader.save_pickle(str(output_path / "label_ids.pickle"), label_ids)
            # TODO check to save metrics
            self.file_loader.save_pickle(str(output_path / "metrics.pickle"), metrics)
            self.file_loader.save_pickle(str(output_path / "test_dataset.pickle"), test_dataset_dict[task_name])
