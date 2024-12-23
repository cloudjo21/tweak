import torch
import pyarrow as pa
import pyarrow.parquet as pq

from pydantic import BaseModel
from typing import Optional

from tunip.iter_utils import chunked_iterators
from tunip.service_config import ServiceLevelConfig

from tweak.predict.predictor import PredictorConfig
from tweak.vector.write import VectorBatch, VectorWriter
from tweak.vector.write.parquet_vector_id_mapping_builder import ParquetVectorIdMappingBuilder


class ParquetRecordVectorWriteSchemaInvalidateException(Exception):
    pass


class ParquetVectorBatch(BaseModel, VectorBatch):
    batch: dict


class ParquetRecordVectorWriter(VectorWriter):
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
        super(ParquetRecordVectorWriter, self).__init__(service_config, task_config, predictor_config, schema, id_field, content_fields, vectors_path, batch_size, build_id_mapping)

        if service_config.has_local_fs:
            self.sink = pa.OSFile(self.vectors_arrow_path, "wb")
            self.file_writer = pa.ipc.new_file(self.sink, self.schema)
        elif service_config.has_hdfs_fs or service_config.has_gcs_fs:
            self.sink = self.file_service.pa_fs.open_output_stream(self.vectors_arrow_path)
            self.file_writer = pq.ParquetWriter(self.sink, self.schema)
        else:
            raise Exception(f"Not supported file system: {service_config.filesystem}")

    def write(self, docs: dict):
        if not self.contain_schema_fields(docs):
            raise ParquetRecordVectorWriteSchemaInvalidateException(f"input documents are not met the following fields of schema: {self.id_field}, {', '.join(self.content_fields)}")

        vec_batch: ParquetVectorBatch = self._get_batch(docs)
        table = pa.table(vec_batch.batch, schema=self.schema)
        self.file_writer.write_table(table)

    def _get_batch(self, batch: dict) -> ParquetVectorBatch:
        d_dict = {}
        for d_name, d_type in zip(self.schema.names, self.schema.types):
            if d_name in self.content_fields:
                continue
            d_dict[d_name] = batch[d_name]

        for content_key in self.content_fields:
            texts = batch[content_key]

            # record document vectors
            embeddings = []
            for chunk in chunked_iterators(iter(texts), len(texts), self.batch_size):
                chunks = list(chunk)
                tensor = self.predictor.predict(chunks)
                embeddings.extend(tensor.to(torch.float32).tolist())

            d_dict[content_key] = embeddings
        return ParquetVectorBatch(batch=d_dict)

    def close(self):
        self.file_writer.close()
        self.sink.close()

        if self.build_id_mapping:
            # build vector id mapping
            self.id_map_builder = ParquetVectorIdMappingBuilder(
                service_config=self.service_config,
                task_config=self.task_config,
                id_field=self.id_field,
                vectors_dir_path=self.vectors_dir_path
            )
            self.id_map_builder.build()
