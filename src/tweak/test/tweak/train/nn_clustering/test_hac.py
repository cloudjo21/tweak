import unittest
import re
import timeit

from sklearn.feature_extraction.text import CountVectorizer

from tunip.es_utils import init_elastic_client
from tunip.nugget_api import Nugget, NuggetFilterResultFormat
from tunip.service_config import get_service_config

from tweak.clustering import NncRequest
from tweak.clustering.factory import NncFactory
from tweak.clustering.distance.bigram_jaccard_dist import (
    BigramJaccardDistanceCalc,
    JaccardDistanceCalcRequest,
    JaccardDistanceCalcResponse,
)


class JaccardDistTest(unittest.TestCase):

    def setUp(self):
        self.index_name = 'news-2022-v1'

        self.re_norm_nums = re.compile('\d+')
        self.es = init_elastic_client(get_service_config())
        self.es.indices.exists(self.index_name)

        self.nugget = Nugget()
        self.white_ptags = ['V', 'N', 'J', 'M', 'SL', 'SH', 'SN']

        self.distance_calc = BigramJaccardDistanceCalc()

        self.max_rows = [1, 2, 3, 4, 5, 7, 10, 100, 150, 200]

    def test_hac_of_nnc(self):

        for max_row in self.max_rows:
            search_res_size = max_row

            print(f"#### {max_row} number of search results ...")

            query_example = "content(넷플릭스) AND content(규제)"
            query = {
                "query": {
                    "query_string": {
                    "query": query_example 
                    }
                },
                "size": search_res_size
            }
            es_results = self.es.search(body=query, index=self.index_name)

            titles = []
            es_doc_ids = []
            for result in es_results['hits']['hits']:
                titles.append(self.re_norm_nums.sub('N', result["_source"]["title"]))
                es_doc_ids.append(result['_id'])


            bigrams = self.nugget.bigrams(
                titles, white_tags=self.white_ptags, result_format=NuggetFilterResultFormat.NUGGET_B_E_LEX
            )
            assert len(bigrams) > 0 and len(bigrams[0]) > 0

            corpus = [' '.join(map('_'.join, bigram_tokens)) for bigram_tokens in bigrams]
            vectorizer = CountVectorizer()

            # scipy.sparse._csr.csr_matrix
            xx = vectorizer.fit_transform(corpus)
            # numpy.ndarray
            import numpy as np
            term_vector = xx.toarray().astype(np.int8)

            dist_calc_req = JaccardDistanceCalcRequest(term_vector= term_vector, num_rows=max_row) # TODO set config from deploy
            start = timeit.default_timer()
            dist_calc_res: JaccardDistanceCalcResponse = self.distance_calc(dist_calc_req)
            duration_in_ms = timeit.default_timer() - start
            print(f"dist calc took {duration_in_ms * 1000} ms.")

            # < 50ms for 100 es result docs
            # < 100ms for 150 es result docs
            # < 200ms for 200 es result docs
            assert duration_in_ms < 0.200

            hac = NncFactory.create("HAC", 0.666666)
            nnc_req = NncRequest(distances=dist_calc_res.distances, dist_calc_status=dist_calc_res.status)
            docid_clusters = hac(nnc_req)

            print(docid_clusters)
