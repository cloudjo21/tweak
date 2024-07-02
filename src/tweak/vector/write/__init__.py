import pyarrow as pa

from abc import ABC, abstractmethod
from functools import reduce
from typing import Optional

from tunip.file_utils import services as file_services
from tunip.path.warehouse import (
    VECTORS_ARROW_FILEPATH,
    WarehouseVectorsTaskPhaseSnapshotPath,
    WarehouseVectorsTaskPhaseSnapshotArrowPath,
    WarehouseVectorsTaskPhaseSnapshotDid2vidPath,
    WarehouseVectorsTaskPhaseSnapshotVid2didPath,
)
from tunip.path_utils import services as path_services
from tunip.service_config import ServiceLevelConfig
from tunip.snapshot_utils import snapshot_now

from tweak.predict.predictor import PredictorConfig
from tweak.predict.predictors import PredictorFactory


class VectorBatch:
    pass


class VectorWriter(ABC):

    @abstractmethod
    def _get_batch(self, batch: dict) -> VectorBatch:
        """
        column-based data
        """
        pass

    @abstractmethod
    def write(self, docs: dict):
        pass

    @abstractmethod
    def close(self):
        pass

    def __init__(
        self,
        service_config: ServiceLevelConfig,
        task_config: dict,
        predictor_config: PredictorConfig,
        schema: pa.Schema,
        id_field: str,
        content_fields: list,
        vectors_path: Optional[str]=None,
        batch_size: int=16,
        build_id_mapping: bool=True,
    ):
        self.service_config = service_config
        self.task_config = task_config

        # pyarrow.Schema
        self.schema: pa.Schema = schema
        self.id_field = id_field
        self.content_fields = content_fields
        self.batch_size = batch_size
        self.snapshot_dt = snapshot_now()

        self.build_id_mapping = build_id_mapping

        self.predictor = PredictorFactory.create(
            predictor_config
        )
        self.file_service = file_services(service_config)
        path_service = path_services(service_config)

        if vectors_path is None:
            vectors_snapshot_path = WarehouseVectorsTaskPhaseSnapshotPath(
                user_name=self.service_config.username,
                task_name=self.task_config["egress"]["task_name"],
                phase_type=self.task_config["egress"]["phase_type"],
                snapshot_dt=self.snapshot_dt
            )
            self.vectors_dir_path = repr(vectors_snapshot_path)
            vectors_arrow_dir_path = repr(WarehouseVectorsTaskPhaseSnapshotArrowPath.from_parent(self.vectors_dir_path))
        else:
            self.vectors_dir_path = vectors_path
            vectors_arrow_dir_path = f"{WarehouseVectorsTaskPhaseSnapshotArrowPath.from_parent_path(parent_path=self.vectors_dir_path)}"

        self.vectors_arrow_path = f"{path_service.build(vectors_arrow_dir_path)}/{VECTORS_ARROW_FILEPATH}"
        self.file_service.mkdirs(str(vectors_arrow_dir_path))

    def contain_schema_fields(self, docs: dict):
        contain_content_fields = reduce(lambda a, b: a and b, [f_name in docs for f_name in self.content_fields])
        if (self.id_field in docs) and contain_content_fields:
            return True
        else:
            return False


class VectorIdMappingBuilder:

    def __init__(self, service_config, task_config, id_field, vectors_dir_path, file_extension):
        self.service_config = service_config
        self.task_config = task_config

        self.file_service = file_services(self.service_config)

        self.task_name = self.task_config['egress'].get('task_name', None)
        self.phase_type = self.task_config["egress"]["phase_type"]

        self.item2vec_id_path = WarehouseVectorsTaskPhaseSnapshotDid2vidPath.from_parent_path(vectors_dir_path)
        self.vec2item_id_path = WarehouseVectorsTaskPhaseSnapshotVid2didPath.from_parent_path(vectors_dir_path)

        self.id_field = id_field

        if vectors_dir_path is None:
            assert self.task_name

            self.snapshot_dt = self.task_config['egress'].get(
                'snapshot_dt', snapshot_now())
            vectors_dir_path = WarehouseVectorsTaskPhaseSnapshotArrowPath(
                user_name=self.service_config.username,
                task_name=self.task_name,
                phase_type=self.phase_type,
                snapshot_dt=self.snapshot_dt
            )
            self.vectors_arrow_path = f"{vectors_dir_path}/{file_extension}"
        else:
            self.file_service.mkdirs(str(vectors_dir_path))
            vectors_dir_path = path_services(service_config).build(vectors_dir_path)
            self.vectors_arrow_path = f"{WarehouseVectorsTaskPhaseSnapshotArrowPath.from_parent_path(parent_path=vectors_dir_path)}/{file_extension}"

    @abstractmethod
    def _open_reader(self):
        pass

    @abstractmethod
    def _map_item2vec_id(self, reader):
        pass

    def build(self):
        source, reader = self._open_reader()

        item2vec_dict = self._map_item2vec_id(reader)

        _ = self._map_vec2item_id(item2vec_dict)

        source.close()

    def _map_vec2item_id(self, item2vec_dict):
        vec2item_dict = dict(
            map(lambda e: (e[1], e[0]), item2vec_dict.items()))

        self.file_service.mkdirs(str(self.vec2item_id_path))
        self.file_service.dumps_pickle(
            f"{self.vec2item_id_path}/vec2item_id.pkl",
            vec2item_dict
        )

        return vec2item_dict
