# TODO DEPRECATED
import io

import pyarrow as pa
import pyarrow.parquet as pq

from tunip.file_utils import services as file_services
from tunip.path.warehouse import (
    VECTORS_ARROW_FILEPATH,
    WarehouseVectorsTaskPhaseSnapshotDid2vidPath,
    WarehouseVectorsTaskPhaseSnapshotVid2didPath,
)

from tweak import LOGGER
from tweak.vector.write import VectorIdMappingBuilder


class ParquetVectorIdMappingBuilder:

    def __init__(self, service_config, task_config, id_field, vectors_dir_path=None):
        # super(ParquetVectorIdMappingBuilder, self).__init__(service_config, task_config, id_field, vectors_dir_path, file_extension)
        self.service_config = service_config
        self.task_config = task_config

        self.file_service = file_services(self.service_config)

        self.task_name = self.task_config['egress'].get('task_name', None)
        self.phase_type = self.task_config["egress"]["phase_type"]

        self.item2vec_id_path = WarehouseVectorsTaskPhaseSnapshotDid2vidPath.from_parent_path(vectors_dir_path)
        self.vec2item_id_path = WarehouseVectorsTaskPhaseSnapshotVid2didPath.from_parent_path(vectors_dir_path)

        self.id_field = id_field


    # def _open_reader(self):
    #     if self.service_config.has_local_fs:
    #         source = pa.OSFile(self.vectors_arrow_path, "rb")
    #         reader = pa.ipc.open_file(source)
    #     elif self.service_config.has_hdfs_fs or self.service_config.has_gcs_fs:
    #         source = self.file_service.pa_fs.open_input_file(self.vectors_arrow_path)
    #         reader = pq.read_table(source, filesystem=self.file_service.pa_fs)

    #     return source, reader

    # def _map_item2vec_id(self, reader):
    #     item2vec_dict = {}
    #     dupl_item2vec_dict = {}
    #     vec_id = 0

    #     LOGGER.info(f"reader.num_rows: {reader.num_rows}")
    #     for i in range(0, reader.num_rows):
    #         id_value = reader.column(self.id_field)[i].as_py()
    #         # row = {self.id_field: reader.column(self.id_field)[i].as_py()}
    #         if id_value in item2vec_dict:
    #             dupl_item2vec_dict[id_value] = vec_id
    #             continue
    #         item2vec_dict[id_value] = vec_id
    #         vec_id = vec_id + 1
    #         if vec_id % 10000 == 0:
    #             LOGGER.info(f"iterating {vec_id} number of items now ...")

    #     LOGGER.info(f"complete to iterate {vec_id} number of items!")

    #     self.file_service.mkdirs(str(self.item2vec_id_path))
    #     self.file_service.dumps_pickle(
    #         f"{self.item2vec_id_path}/item2vec_id.pkl",
    #         item2vec_dict
    #     )
    #     self.file_service.dumps_pickle(
    #         f"{self.item2vec_id_path}/dupl_item2vec_id.pkl",
    #         dupl_item2vec_dict
    #     )

    #     return item2vec_dict

    def build_with_sdf(self, vector_sdf):
        # TODO build id mappings corresponding to vector-index-building
        iid2vid_pdf = vector_sdf.filter(f"{self.id_field} != ''").select(self.id_field, "vid").toPandas()

        self.file_service.mkdirs(str(self.item2vec_id_path))
        i2v_dict = dict([(x[self.id_field], x["vid"]) for x in iid2vid_pdf[[self.id_field, "vid"]].to_dict(orient="records")])
        self.file_service.dumps_pickle(f"{str(self.item2vec_id_path)}/item2vec_id.pkl", i2v_dict)
        i2v_dict = None

        self.file_service.mkdirs(str(self.vec2item_id_path))
        v2i_dict = dict([(x["vid"], x[self.id_field]) for x in iid2vid_pdf[["vid", self.id_field]].to_dict(orient="records")])
        self.file_service.dumps_pickle(f"{str(self.vec2item_id_path)}/vec2item_id.pkl", v2i_dict)
        v2i_dict = None

        iid2vid_pdf = None
