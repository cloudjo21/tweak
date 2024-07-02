from tunip.config import Config

from tweak.metrics.task2metric import MetricBuilder
from tweak.utils.task_set_yaml_parser import TaskSetYamlParser

import os

service_stage_config_path = f'{os.environ["NAUTS_HOME"]}/experiments/ed/resources/application.json'
task_config_path = f'{os.environ["NAUTS_HOME"]}/experiments/ed/resources/ner/train/nc_ner/train.yml'

config = Config(service_stage_config_path)
task_set_parser = TaskSetYamlParser(yaml_file=task_config_path, config=config)
task_set = task_set_parser.parse()


task = next(iter(task_set))
print(task.task_type)
print(type(task.task_type))

print(list(MetricBuilder.predifined_metrics.keys())[0])
print(type(list(MetricBuilder.predifined_metrics.keys())[0]))


print(task.task_type in MetricBuilder.predifined_metrics.keys())

from enum import Enum
class Color(Enum):
    RED = 1
    BLUE = 2

c_entries = {Color.RED: 111, Color.BLUE: 222}
print(Color.RED in c_entries)

compute_metrics_dict = MetricBuilder.create(
    taskset=task_set,
    label_list_dict={}
)
