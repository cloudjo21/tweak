from dataclasses import dataclass
from itertools import filterfalse
from typing import List, Optional

from tweak.task.task_set import TaskType


@dataclass
class EvalTask:
    task_name: str
    task_type: TaskType
    target_corpus_dt: str
    eval_batch_size: int
    max_length: int
    checkpoint: str


class EvalTaskSet:
    def __init__(self, tasks: List[EvalTask], domain_name: str, snapshot_dt: str):
        self.tasks = tasks
        self.domain_name = domain_name
        self.snapshot_dt = snapshot_dt

    def __iter__(self):
        return iter(self.tasks)
    
    def __getitem__(self, name):
        filtered = list(filter(lambda x: x.task_name == name, self.tasks))
        assert len(filtered) < 2
        return filtered[0] if filtered else None

    def get_by_name(self, name: str) -> Optional[EvalTask]:
        filtered_tasks = list(filterfalse(lambda t: t.task_name != name, self.tasks))

        assert len(filtered_tasks) < 2

        if len(filtered_tasks) > 0:
            return filtered_tasks[0]
        return None

    @property
    def names(self):
        return [task.task_name for task in self.tasks]

    @property
    def types(self):
        return [task.task_type for task in self.tasks]
