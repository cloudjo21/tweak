import unittest
import urllib.parse

from tunip.path.mart import MartPretrainedModelPath
from tunip.path_utils import TaskPath
from tunip.service_config import get_service_config
from tunip.snapshot_utils import SnapshotPathProvider, snapshot_now

from tweak.predict.predictor import PredictorConfig
from tweak.predict.predictors import PredictorFactory
from tweak.vector.fetch.top_k_vector_fetcher import TopKVectorFetcher
from tweak.vector.supply import GenerationVectorSupplier
from tweak.vector.write.search_index_vector_writer import SearchIndexVectorWriter


class TopKVectorFetcherTest(unittest.TestCase):

    def setUp(self):
        # TODO define predictor_config including "predict_output_type": "last_hidden.global_mean_pooing"
        # TODO and ready to serve prediction_build_cls for "last_hidden.global_mean_pooing" for PredictionToolboxPackerForSeq2SeqLM
        domain_name = 'query2item_intro'
        snapshot_dt = '20230222_140415_173430'
        # checkpoint = 'checkpoint-3500'
        task_name = 'generation'

        service_config = get_service_config()
        self.task_config = {
            "ingress": {
                "domain_name": "query2item_intro",
                "task_name": task_name,
                "source_type": "corpus.tjsonl",
                "path_type": "document",
                "doc_field_mappings": [
                    {"out_field": "text", "in_fields": ["text"]}
                ],
                "group_fields": ["item_id", "specialty"],
                "input_tokens_field": "reco_request_tokens",
            },
            "egress": {
                "path_type": "document",
                "task_name": task_name,
                "source_type": "corpus.tjsonl",
                "snapshot_dt": snapshot_now()
            }
        }

        self.INDEX_NAME = f"vec_{domain_name}"

        # /data/home/ed/temp/user/ed/domains/query2item_intro/20230217_220119_168166/models/checkpoint-3500/generate
        # model_path = ModelPath(service_config.username, domain_name, snapshot_dt, task_name, checkpoint)
        model_path = TaskPath(service_config.username, domain_name, snapshot_dt, task_name)

        model_name = "hyunwoongko/kobart"
        quoted_model_name = urllib.parse.quote(model_name, safe='')
        plm_model_path = MartPretrainedModelPath(
            user_name=service_config.username,
            model_name=quoted_model_name
        )
        tokenizer_path = str(plm_model_path) + "/vocab"

        self.pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "torchscript",
            "predict_output_type": "last_hidden.global_mean_pooling",
            "device": "cuda",
            "model_config": {
                "model_path": str(model_path),
                "model_name": model_name,
                "task_name": task_name,
                "task_type": "SEQ2SEQ_LM",
            },
            "tokenizer_config": {
                "model_path": str(plm_model_path),
                "path": tokenizer_path,
                "max_length": 64
            }
        }
        self.predict_config = PredictorConfig.model_validate(self.pred_config_json)
        print(self.predict_config)
        print(type(self.predict_config))

    def test_fetch(self):
        service_config = get_service_config()

        vec_supplier = GenerationVectorSupplier(predictor=PredictorFactory.create(self.predict_config))
        index_mapping = {
            "mappings": {
                "properties": {
                    "reco_request_tokens": {
                        "type": "text"
                    },
                    "career_intro_tokens": {
                        "type": "text"
                    },
                    "interaction": {
                        "type": "integer"
                    },
                    "passion": {
                        "type": "integer"
                    },
                    "promise": {
                        "type": "integer"
                    },
                    "sid": {
                        "type": "keyword"
                    },
                    "item_id": {
                        "type": "keyword"
                    },
                    "intro_vector": {
                        "type": "dense_vector",
                        "dims": 768
                    }
                }
            }
        }
        text_fields = ["text", "next_text"]
        # vec_fields = []
        vec_writer = SearchIndexVectorWriter(service_config, vec_supplier, self.INDEX_NAME, index_mapping, text_fields)
        task = TopKVectorFetcher(service_config, self.task_config)
        task.fetch(vec_writer)
