import sys

from datetime import datetime, timedelta, timezone
from pyspark.sql.functions import first
from typing import Optional

from tunip.file_utils import services as file_services
from tunip.path_utils import services as path_services
from tunip.snapshot_utils import SnapshotPathProvider
from tunip.spark_utils import SparkConnector
from tunip.time_it import time_it

from tweak.vector.write import VectorWriter
from tweak import LOGGER


class VectorFetcher:

    def __init__(self, service_config, task_config):
        self.service_config = service_config
        self.task_config = task_config
        self.n_batch = 256

        self.task_name = self.task_config['ingress'].get('task_name', None)
        self.domain_name = self.task_config['ingress']['domain_name']
        self.group_by_fields = self.task_config['ingress'].get('group_by_fields', None)
        self.group_select_field = self.task_config['ingress'].get('group_select_field', None)

        self.select_fields = self.task_config['egress'].get('select_fields', None)
        self.max_records = self.task_config['egress'].get("max_records", sys.maxsize)

        self.file_service = file_services.get(
            service_config.filesystem.upper(),
            config=service_config.config
        )
        self.path_service = path_services.get(
            service_config.filesystem.upper(),
            config=service_config.config
        )

    def _len_of_batch(self, doc_batch):
        first_key = next(iter(doc_batch), None)
        if first_key:
            return len(doc_batch[first_key])
        else:
            return 0

    @time_it(LOGGER)
    def fetch(self, writer: VectorWriter, ingress_latest_path: Optional[str]=None, format: Optional[str]="json"):

        # init. ingress path
        if ingress_latest_path is None:
            assert self.task_name is not None
            ingress_path = f"/user/{self.service_config.username}/warehouse/vector_input/{self.task_name}/{self.domain_name}"
            snapshot_path_provider = SnapshotPathProvider(self.service_config)
            ingress_latest_path = snapshot_path_provider.latest(ingress_path)

        spark = SparkConnector.getOrCreate(local=True)
        # TODO stream input/output
        assert format in ["json", "parquet"]
        df = spark.read.format(format).load(ingress_latest_path)
        
        if self.group_by_fields and self.group_select_field:
            df = df.groupBy(self.group_by_fields).agg(first(self.group_select_field).alias(self.group_select_field)).distinct()
        schema_keys = []
        for f in df.schema.fields:
            if not self.select_fields or (f.name in self.select_fields):
                schema_keys.append(f.name)
        rows = df.collect()

        print(f"{len(rows)} number of rows are read from {ingress_latest_path}")

        count = 0
        doc_batch = dict()
        for k in schema_keys:
            doc_batch[k] = []

        for row in rows:
            doc = row.asDict()

            if self._len_of_batch(doc_batch) < self.n_batch:

                # the column names are dependent to the schema type of document (lake/document/[schema_type])
                # TODO diversify column schema according to data source(path_type, source_type, ...)

                for k in schema_keys:
                    doc_batch[k].append(doc[k])
            else:
                t = datetime.now(timezone(timedelta(hours=9)))

                writer.write(doc_batch)

                for k in schema_keys:
                    doc_batch[k] = []
                    doc_batch[k].append(doc[k])

                print(f"{datetime.now(timezone(timedelta(hours=9)))-t} secs...")
                print(f"{count} data are written ...")

            count += 1

            if count > self.max_records:
                print("count > self.max_records")
                break

        if self._len_of_batch(doc_batch) > 0:
            t = datetime.now(timezone(timedelta(hours=9)))

            writer.write(doc_batch)

            print(f"{datetime.now(timezone(timedelta(hours=9)))-t} secs...")
            print(f"{count} data are written ...")

        writer.close()

        return 0
