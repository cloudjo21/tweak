import json
import sys

from pyspark.sql.functions import collect_list

from tunip.corpus_utils_v2 import CorpusToken, merge_surface
from tunip.file_utils import services as file_services
from tunip.path_utils import services as path_services
from tunip.snapshot_utils import SnapshotPathProvider, snapshot_now
from tunip.spark_utils import SparkConnector
from tunip.time_it import time_it

from tweak import LOGGER
from tweak.vector.write import VectorWriter


class TopKVectorFetcher:

    def __init__(self, service_config, task_config):
        self.service_config = service_config
        self.task_config = task_config
        self.n_batch = 256

        self.max_records = self.task_config['egress'].get("max_records", sys.maxsize)
        self.valid_gte_length = self.task_config['ingress'].get("valid_gte_length", [])
	self.item_id = self.task_config["ingress"].get("item_id", None)
        
        """
        - doc_field_mappings:
            - out_field: TITLE
              in_fields: MAIN_TITLE, SUB_TITLE
            - out_field: TEXT
              in_fields: SUMMARY, CONTENT
        """
        # self.doc_field_mappings = self.task_config['ingress'].get('doc_field_mappings', [])
        # self.id_field = self.task_config['ingress'].get('id_field', '_id')

        self.file_service = file_services.get(
            service_config.filesystem.upper(),
            config=service_config.config
        )
        self.path_service = path_services.get(
            service_config.filesystem.upper(),
            config=service_config.config
        )

    def _len_of_batch(self, doc_batch):
        if doc_batch.keys():
            return len(doc_batch[doc_batch.keys()[0]])
        else:
            return 0

    @time_it(LOGGER)
    def fetch(self, writer: VectorWriter):

        self.task_config['egress']['snapshot_dt'] = self.task_config['egress'].get('snapshot_dt', snapshot_now())

        # init. ingress path
        task_name = self.task_config['ingress']['task_name']
        domain_name = self.task_config['ingress']['domain_name']
        group_fields = self.task_config['ingress']['group_fields']
        input_tokens_field = self.task_config['ingress']['input_tokens_field']

        ingress_path = f"/user/{self.service_config.username}/warehouse/top_k_vector_input/{task_name}/{domain_name}"
        snapshot_path_provider = SnapshotPathProvider(self.service_config)
        ingress_path = snapshot_path_provider.latest(ingress_path)

        spark = SparkConnector.getOrCreate(local=True)
        # TODO stream input/output
        # item_id, specialty, request, intro
        rows = spark.read.json(ingress_path).where(f"{self.item_id}!=null") \
            .select(*(group_fields + [input_tokens_field])) \
            .groupBy(*group_fields).agg(
                collect_list(input_tokens_field).alias(f'{input_tokens_field}_list')
            )

        print(f"{len(rows)} number of rows are read from {ingress_path}")

        count = 0
        doc_batch = dict()
        for k in rows.schema.keys():
            doc_batch[k] = []

        for row in rows:
            # chunk of 'input_*_field' grouped by 'group_fields'
            chunk = row.asDict()

            texts = []
            for tokens_str in chunk[input_tokens_field]:
                texts.append(merge_surface([CorpusToken.model_validate(t) for t in json.loads(tokens_str)]))
            input_texts_field = input_tokens_field.replace('_tokens_list', '_texts')
            chunk.update({input_texts_field: '\n'.join(texts)})

            writer.write_chunk(chunk)

            count += 1
            if count % 1000 == 0:
                print(f"{count} data are written ...")

            if count > self.max_records:
                print(f"#-of-record:{count} > maximum-of-record:{self.max_records}")
                break

        writer.close()

        return 0
