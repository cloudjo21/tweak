import pyarrow as pa

from pydantic import BaseModel
from typing import Optional

from tunip.iter_utils import chunked_iterators
from tunip.service_config import ServiceLevelConfig

from tweak.predict.predictor import PredictorConfig
from tweak.vector.write import VectorBatch, VectorWriter
from tweak.vector.write.arrow_vector_id_mapping_builder import ArrowVectorIdMappingBuilder


class ArrowRecordVectorWriteSchemaInvalidateException(Exception):
    pass


class ArrowVectorBatch(BaseModel, VectorBatch):
    batch: pa.RecordBatch


class ArrowRecordVectorWriter(VectorWriter):
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
        build_id_mapping: bool=True
    ):

        super(ArrowRecordVectorWriter, self).__init__(service_config, task_config, predictor_config, schema, id_field, content_fields, vectors_path, batch_size, build_id_mapping)

        if service_config.has_local_fs:
            self.sink = pa.OSFile(self.vectors_arrow_path, "wb")
            self.file_writer = pa.ipc.new_file(self.sink, self.schema)
        elif service_config.has_hdfs_fs or service_config.has_gcs_fs:
            hdfs = self.file_service.pa_fs
            self.sink = hdfs.open_output_stream(self.vectors_arrow_path)
            self.file_writer = pa.ipc.new_file(self.sink, self.schema)
        else:
            raise Exception(f"Not supported file system: {service_config.filesystem}")

    def write(self, docs: dict):
        if not self.contain_schema_fields(docs):
            raise ArrowRecordVectorWriteSchemaInvalidateException(f"input documents are not met the following fields of schema: {self.id_field}, {', '.join(self.content_fields)}")

        vec_batch: ArrowVectorBatch = self._get_batch(docs)
        self.file_writer.write(vec_batch.batch)

    def _get_batch(self, batch: dict) -> VectorBatch:
        content_arr_list = []

        d_names = []
        d_arrays = []
        for d_name, d_type in zip(self.schema.names, self.schema.types):
            if d_name in self.content_fields:
                continue
            d_names.append(d_name)
            d_arrays.append(pa.array(batch[d_name], type=d_type))

        for content_key in self.content_fields:
            texts = batch[content_key]

            # record document vectors
            embeddings = []
            for chunk in chunked_iterators(iter(texts), len(texts), self.batch_size):
                chunks = list(chunk)
                tensor = self.predictor.predict(chunks)
                embeddings.extend(tensor.tolist())

            h_vec_arr = pa.array(embeddings, type=pa.list_(pa.float32()))
            content_arr_list.append(h_vec_arr)
        
        return ArrowVectorBatch(batch=pa.record_batch(d_arrays + content_arr_list, names=d_names + self.content_fields))

    def close(self):
        self.file_writer.close()
        self.sink.close()

        if self.build_id_mapping:
            # build vector id mapping
            self.id_map_builder = ArrowVectorIdMappingBuilder(
                service_config=self.service_config,
                task_config=self.task_config,
                id_field=self.id_field,
                vectors_dir_path=self.vectors_dir_path
            )
            self.id_map_builder.build()
