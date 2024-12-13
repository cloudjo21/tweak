from transformers.training_args import TrainingArguments

from tunip.config import Config
from tunip.env import NAUTS_LOCAL_ROOT
from tunip.path_utils import default_local_user_dir
from tunip.snapshot_utils import snapshot_now
from tunip.yaml_loader import YamlLoader

from tweak.task.pretraining_task_set import InverseClozeTask
from tweak.task.task_set import InputColumn, InputColumnType, Task, TaskSet, TaskType


class NoInputColumnsException(Exception):
    pass


class TaskSetYamlParser(YamlLoader):
    REQUIRED_ATTR_TASK_SET_FOR_DOMAIN = [
        "service_repo_dir",
        "user_name",
        "domain_name",
        "resume_from_checkpoint",
        "snapshot_dt",
    ]
    REQUIRED_ATTR_TASK_SET_FOR_TRAINING_ARGS = [
        "pretrained_model_name",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "eval_accumulation_steps",
        "learning_rate",
        "num_train_epochs",
        "evaluation_strategy",
        "save_strategy",
        "save_steps",
        "eval_strategy",
        "eval_steps",
        "eval_delay",
        "logging_steps",
        "save_total_limit",
        "metric_for_best_model",
        "weight_decay",
        "seed",
        "local_rank",
        "label_smoothing_factor",
        "warmup_ratio",
        "warmup_steps"
    ]

    def __init__(self, yaml_file: str, config: Config):
        super(TaskSetYamlParser, self).__init__(yaml_file)
        self.service_stage_config = config


    def parse(self) -> TaskSet:
        cfg = super().load()
        task_set_dict = cfg["TASK_SET"]
        service_repo_dir = task_set_dict.get("service_repo_dir") or (
            default_local_user_dir(self.service_stage_config) or NAUTS_LOCAL_ROOT
        )

        # THIS IS FOR THE USER OF SERVICE REPOSITORY (NOT the /home/[username] of local machine)
        # and this parameter would provide the training execution to indivisual user
        user_name = task_set_dict.get("user_name") or "nauts"
        domain_name = task_set_dict["domain_name"]
        # TODO if 'snapshot_dt' is given, then we would support to resume training in the future ...
        run_session = task_set_dict.get("snapshot_dt") or snapshot_now()
        pretrained_model_name = task_set_dict["pretrained_model_name"]
        label_names = task_set_dict.get("label_names", ["labels"])
        # label_names = ["labels"]

        resume_from_checkpoint = task_set_dict.get("resume_from_checkpoint") or None

        model_output_dir = self._get_output_dir_with_snapshot(
            service_repo_dir=service_repo_dir,
            user_name=user_name,
            domain_name=domain_name,
            snapshot_dt=run_session
        )

        training_args = TrainingArguments(
            output_dir=model_output_dir,
            overwrite_output_dir=True,
            do_train=True,
            eval_accumulation_steps=task_set_dict.get("eval_accumulation_steps") or None,
            learning_rate=float(task_set_dict.get("learning_rate") or 1e-5),
            per_device_train_batch_size=task_set_dict.get("per_device_train_batch_size") or 8,
            per_device_eval_batch_size=task_set_dict.get("per_device_eval_batch_size") or 8,
            evaluation_strategy=task_set_dict.get("evaluation_strategy") or "epoch",
            save_strategy=task_set_dict.get("save_strategy") or "epoch",
            # use if evaluation_strategy is 'epoch'
            num_train_epochs=task_set_dict.get("num_train_epochs") or 3,
            # use if evaluation_strategy is 'steps'
            eval_steps=task_set_dict.get("eval_steps") or 500,
            eval_delay=task_set_dict.get("eval_delay") or 5,
            save_steps=task_set_dict.get("save_steps") or 1500,
            logging_steps=task_set_dict.get("logging_steps") or 100,
            save_total_limit=task_set_dict.get("save_total_limit") or 3,
            metric_for_best_model=task_set_dict.get("metric_for_best_model") or None,
            load_best_model_at_end=False,
            ignore_data_skip=False,
            weight_decay=task_set_dict.get("weight_decay") or 0,
            seed=task_set_dict.get("seed") or 42,
            local_rank=task_set_dict.get("local_rank") or -1,
            label_names=label_names,
            label_smoothing_factor=task_set_dict.get("label_smoothing_factor") or 0.0,
            warmup_ratio=task_set_dict.get("warmup_ratio") or 0,
            warmup_steps=task_set_dict.get("warmup_steps") or 0,
            optim=task_set_dict.get('optim') or 'adamw_hf',
        )

        new_task_list = []
        for key, item in task_set_dict.items():
            if (key not in self.REQUIRED_ATTR_TASK_SET_FOR_DOMAIN) and (
                key not in self.REQUIRED_ATTR_TASK_SET_FOR_TRAINING_ARGS
            ):
                name, task = key, item
                if task["task_type"].upper() == 'INVERSE_CLOZE_TASK':
                    new_task = InverseClozeTask(
                        task_name=name,
                        task_type=TaskType[task["task_type"].upper()],
                        dataset_name=task["dataset_name"],
                        dataset_path=task["dataset_path"],
                        input_columns=self._get_input_columns(task),
                        label_column_name=task.get("label_column_name") or None,
                        label_names=task.get('label_names') or None,
                        pretrained_model_name=pretrained_model_name,
                        max_length=task.get("max_length") or 32
                    )
                else:
                    new_task = Task(
                        task_name=name,
                        task_type=TaskType[task["task_type"].upper()],
                        dataset_name=task["dataset_name"],
                        dataset_path=task["dataset_path"],
                        input_columns=self._get_input_columns(task),
                        label_column_name=task.get("label_column_name") or None,
                        label_names=task.get('label_names') or None,
                        pretrained_model_name=pretrained_model_name,
                        max_length=task.get("max_length") or 32,
                        source_bio_from_dataset=task.get('source_bio_from_dataset') or False,
                        problem_type=task.get('problem_type') or None,
                        pad_token=task.get('pad_token') or None,
                        use_vocab_label=task.get('use_vocab_label') or False
                    )
                new_task_list.append(new_task)

        new_task_set = TaskSet(
            tasks=new_task_list,
            service_repo_dir=service_repo_dir,
            user_name=user_name,
            domain_name=domain_name,
            snapshot_dt=run_session,
            training_args=training_args,
            resume_from_checkpoint=resume_from_checkpoint
        )
        return new_task_set
    
    def _get_input_columns(self, task: dict):
        if "input_columns" in task.keys():
            # return task["input_columns"]
            input_columns = []
            for input_column in task["input_columns"]:
                input_columns.append(InputColumn(type_=InputColumnType[input_column["type"].upper()], name=input_column["name"]))
            return input_columns
        else:
            raise NoInputColumnsException()

    def _get_output_dir_with_snapshot(
        self, service_repo_dir, user_name, domain_name, snapshot_dt
    ):
        local_user_home_dir = f"{service_repo_dir}/user/{user_name}"
        snapshot_dir = f"{local_user_home_dir}/domains/{domain_name}/{snapshot_dt}"
        model_output_dir = f"{snapshot_dir}/model"
        return model_output_dir
