import pyarrow as pa

from tunip.path.warehouse import VECTORS_ARROW_FILEPATH

from tweak import LOGGER
from tweak.vector.write import VectorIdMappingBuilder


class ArrowVectorIdMappingBuilder(VectorIdMappingBuilder):

    def __init__(self, service_config, task_config, id_field, vectors_dir_path=None, file_extension=VECTORS_ARROW_FILEPATH):
        super(ArrowVectorIdMappingBuilder, self).__init__(service_config, task_config, id_field, vectors_dir_path, file_extension)

    def _open_reader(self):
        if self.service_config.has_local_fs:
            source = pa.OSFile(self.vectors_arrow_path, "rb")
            reader = pa.ipc.open_file(source)
        elif self.service_config.has_hdfs_fs or self.service_config.has_gcs_fs:
            source = self.file_service.pa_fs.open_input_file(self.vectors_arrow_path)
            reader = pa.ipc.open_file(source)

        return source, reader

    def _map_item2vec_id(self, reader):
        item2vec_dict = {}
        dupl_item2vec_dict = {}
        vec_id = 0

        LOGGER.info(f"reader.num_record_batches: {reader.num_record_batches}")
        for i in range(0, reader.num_record_batches):
            record_batch = reader.get_batch(i).to_pandas().to_dict('records')
            for record in record_batch:
                # assert record[self.id_field] not in item2vec_dict
                if record[self.id_field] in item2vec_dict:
                    dupl_item2vec_dict[record[self.id_field]] = vec_id
                    continue
                item2vec_dict[record[self.id_field]] = vec_id
                vec_id = vec_id + 1
                if vec_id % 10000 == 0:
                    LOGGER.info(f"iterating {vec_id} number of items now ...")

        LOGGER.info(f"complete to iterate {vec_id} number of items!")

        self.file_service.mkdirs(str(self.item2vec_id_path))
        self.file_service.dumps_pickle(
            f"{self.item2vec_id_path}/item2vec_id.pkl",
            item2vec_dict
        )
        self.file_service.dumps_pickle(
            f"{self.item2vec_id_path}/dupl_item2vec_id.pkl",
            dupl_item2vec_dict
        )

        return item2vec_dict
