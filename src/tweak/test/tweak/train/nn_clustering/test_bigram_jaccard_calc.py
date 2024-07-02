import unittest
import re
import timeit

from sklearn.feature_extraction.text import CountVectorizer

from tunip.es_utils import init_elastic_client
from tunip.nugget_api import Nugget, NuggetFilterResultFormat
from tunip.service_config import get_service_config
from tweak.clustering.distance.bigram_jaccard_dist import (
    BigramJaccardDistanceCalc,
    JaccardDistanceCalcRequest,
    JaccardDistanceCalcResponse
)



index_name = 'news-2022-v1'

re_norm_nums = re.compile('\d+')
es = init_elastic_client(get_service_config())
es.indices.exists(index_name)

nugget = Nugget()
white_ptags = ['V', 'N', 'J', 'M', 'SL', 'SH', 'SN']

distance_calc = BigramJaccardDistanceCalc()
# max_row = 200


max_row = search_res_size = 100
hac_dist_threshold = 0.6666

query_example = "content(넷플릭스) AND content(규제)"
query = {
    "query": {
        "query_string": {
        "query": query_example 
        }
    },
    "size": search_res_size
}
es_results = es.search(query, index=index_name)

titles = []
es_doc_ids = []
for result in es_results['hits']['hits']:
    titles.append(re_norm_nums.sub('N', result["_source"]["title"]))
    es_doc_ids.append(result['_id'])


bigrams = nugget.bigrams(
    titles, white_tags=white_ptags, result_format=NuggetFilterResultFormat.NUGGET_B_E_LEX
)
assert len(bigrams) > 0 and len(bigrams[0]) > 0

# print(bigrams[0])
# exit(0)

corpus = [' '.join(map('_'.join, bigram_tokens)) for bigram_tokens in bigrams]
vectorizer = CountVectorizer()

# scipy.sparse._csr.csr_matrix
xx = vectorizer.fit_transform(corpus)
    # numpy.ndarray
term_vector = xx.toarray()

dist_calc_req = JaccardDistanceCalcRequest(term_vector= term_vector, num_rows=max_row) # TODO set config from deploy
start = timeit.default_timer()
dist_calc_res: JaccardDistanceCalcResponse = distance_calc(dist_calc_req)
duration_in_ms = timeit.default_timer() - start
print(f"dist calc took {duration_in_ms} ms.")

assert duration_in_ms < 0.050
