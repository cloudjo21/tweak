import torch

from elasticsearch.helpers import bulk

from tunip.es_utils import init_elastic_client
from tunip.iter_utils import chunked_iterators
from tunip.service_config import ServiceLevelConfig

from tweak.predict.predictor import Predictor
from tweak.vector.write import VectorBatch, VectorWriter


class SearchIndexVectorWriter(VectorWriter):

    def __init__(
        self,
        service_config: ServiceLevelConfig,
        supplier: Predictor,
        index_name: str,
        schema: dict,
        content_fields: list,
        content_vector_fields: list,
        batch_size=8
    ):
        self.service_config = service_config

        self.supplier = supplier

        # index_mapping json
        self.schema: dict = schema
        # self.fields = schema["mappings"]["properties"].keys()
        self.content_fields = content_fields
        self.content_vector_fields = content_vector_fields

        self.index_name = index_name
        self.batch_size = batch_size

        self.search_client = init_elastic_client(service_config)
        self.chunk_batch = []

    def write(self, docs: dict):
        batch = self._get_batch(docs)
        es_docs = list(self._get_docs(batch))
        bulk(self.search_client, es_docs)

    def write_chunk(self, docs: dict):
        chunk = self._get_top_k_chunk(
            chunk=docs,
            input_field=self.content_fields,
            input_vector_field=self.content_vector_fields
        )
        self.chunk_batch.append(chunk)
        if len(self.chunk_batch) == self.batch_size == 0:
            bulk(
                self.search_client,
                self.chunk_batch,
                request_timeout=3,
            )
            self.chunk_batch = []
    
    def _has_indexible_vectors(self):
        return len(self.content_fields) > 0 and (len(self.content_fields) == len(self.content_vector_fields))

    def _get_batch(self, batch) -> VectorBatch:
        # TODO assert len(batch) <= self.batch_size if use_top_k_fetch is True

        if self._has_indexible_vectors():
            content2embeddings = dict()
            for c_key in self.content_vector_fields:
                content2embeddings[c_key] = []

            for key in batch.keys():
                if key not in self.content_fields:
                    continue
                else:
                    c_key = self.content_vector_fields[self.content_fields.index(key)]

                # treat only c_key
                values = batch[key]

                for iter_chunk in chunked_iterators(iter(values), len(values), self.batch_size):
                    chunks = list(iter_chunk)
                    tensor = self.supplier.predict(chunks)
                    content2embeddings[c_key].extend(tensor.tolist())

            batch.update(content2embeddings)
        return batch

    def _get_top_k_chunk(self, chunk, input_fields, input_vector_fields):
        if input_vector_fields is None:
            input_vector_fields = [f"{v}_vec" for v in input_vector_fields]
        assert len(input_fields) == len(input_vector_fields)

        for input_field, input_vector_field in zip(input_fields, input_vector_fields):
            texts = chunk[input_field]
            tensor: torch.FloatTensor = self.supplier.predict(texts)
            chunk.update({input_vector_field: tensor.tolist()})
            # del chunk[input_field]
        return chunk

    def _get_docs(self, docs):
        keys = list(docs.keys())
        for _doc in zip(*[docs[k] for k in keys]):
            doc = dict()
            for k, v in zip(keys, _doc):
                doc.update({k: v})

            doc.update({"_index": self.index_name})
            yield doc

    def close(self):
        if self.chunk_batch:            
            bulk(
                self.search_client,
                self.chunk_batch
            )
            self.chunk_batch = []
        self.search_client.close()
