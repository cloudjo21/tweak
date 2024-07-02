import unittest
import urllib.parse

from tunip.service_config import get_service_config
from tunip.snapshot_utils import snapshot_now
from tunip.path_utils import TaskPath
from tunip.path.mart import MartPretrainedModelPath

from tweak.predict.predictor import PredictorConfig
from tweak.predict.predictors import PredictorFactory
from tweak.vector.fetch.vector_fetcher import VectorFetcher
from tweak.vector.write.search_index_vector_writer import SearchIndexVectorWriter


class LoadVectorFetcherTest(unittest.TestCase):

    def setUp(self):
        service_config = get_service_config()
        task_config = {
            "ingress": {
                "domain_name": "query2item",
                "task_name": "generation",
                "model_snapshot_dt": "20230328_130916_874499",
                "source_type": "corpus.tjsonl",
                "id_field": "sid",
                "path_type": "document",
                "content_fields": ['text', 'next_text']
            },
            "egress": {
                "index_name": "query2item_vectors",
                "index_mapping": {
                    "mappings": {
                        "properties": {
                            "sid": {
                                "type": "keyword"
                            },
                            "item_id": {
                                "type": "keyword"
                            },
                            "specialty": {
                                "type": "integer"
                            },
                            "text": {
                                "type": "text"
                            },
                            "next_text": {
                                "type": "text"
                            },
                            "text_vector": {
                                "type": "dense_vector",
                                "dims": 768,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "next_text_vector": {
                                "type": "dense_vector",
                                "dims": 768
                            },
                            "interaction": {
                                "type": "float"
                            },
                            "passion": {
                                "type": "float"
                            },
                            "promise": {
                                "type": "float"
                            }
                        }
                    }
                },
                "content_vector_fields": ['text_vector', 'next_text_vector'],
                "snapshot_dt": snapshot_now()
            }
        }

        content_fields = task_config['ingress']['content_fields']
        domain_name = task_config['ingress']['domain_name']
        task_name = task_config['ingress']['task_name']
        model_snapshot_dt = task_config['ingress']['model_snapshot_dt']
        # checkpoint = 'checkpoint-3500'

        index_mapping = task_config['egress']['index_mapping']
        index_name = task_config['egress']['index_name']
        content_vector_fields = task_config['egress']['content_vector_fields']

        from tunip.env import NAUTS_LOCAL_ROOT
        # /data/home/ed/temp/user/ed/domains/query2item/20230217_220119_168166/models/checkpoint-3500/generate
        model_path = str(TaskPath(service_config.username, domain_name, model_snapshot_dt, task_name)) + "/torchscript/encoder"
        # model_path = str(ModelPath(service_config.username, domain_name, model_snapshot_dt, task_name, checkpoint)) + "/" + "encoder"
        print(model_path)

        model_name = "hyunwoongko/kobart"
        quoted_model_name = urllib.parse.quote(model_name, safe='')
        plm_model_path = str(MartPretrainedModelPath(
            user_name=service_config.username,
            model_name=quoted_model_name
        ))
        tokenizer_path = str(plm_model_path) + "/vocab"

        pred_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "torchscript",
            "predict_output_type": "last_hidden.mean_pooling",
            "device": "cuda",
            "model_config": {
                "model_path": str(model_path),
                "model_name": model_name,
                "task_name": "generation",
                "task_type": "SEQ2SEQ_LM",
                "encoder_only": True
            },
            "tokenizer_config": {
                "model_path": str(plm_model_path),
                "path": tokenizer_path,
                "max_length": 128
            }
        }
        pred_config_obj = PredictorConfig.model_validate(pred_config_json)

        predictor = PredictorFactory.create(pred_config_obj)
        # TODO get predictor from PredictorFactory for predict_seq2seq_lm_encoder.PredictorSeq2SeqLmEncoder

        # out = vec_supplier(['안녕', '안녕'])
        # print(out.shape)
        # print(out.tolist())
        # exit(0)

        self.vec_writer = SearchIndexVectorWriter(
            service_config=service_config,
            supplier=predictor,
            # supplier=vec_supplier,
            index_name=index_name,
            schema=index_mapping,
            content_fields=content_fields,
            content_vector_fields=content_vector_fields,
            batch_size=16
        )
        self.task = VectorFetcher(service_config, task_config)

    def test_load(self):
        response = self.task.fetch(self.vec_writer)
        assert response == 0
