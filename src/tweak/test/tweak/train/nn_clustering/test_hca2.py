import json
from matplotlib.colors import LinearSegmentedColormap
from sklearn import cluster
from tunip.es_utils import init_elastic_client
from tunip.service_config import get_service_config

index_name = 'news-2022-v1'
search_res_size = 70
hac_dist_threshold = 0.6666

# TODO use preprocess or normalize
import re

# 12 -> N, 98356 -> N
re_norm_nums = re.compile('\d+')

es = init_elastic_client(get_service_config())
es.indices.exists(index_name)

#query_example = "content(넷플릭스) AND content(규제)"
query_example = "content(오징어)"
query = {
  "query": {
    "query_string": {
      "query": query_example 
    }
  },
  "size": search_res_size
}
es_results = es.search(query, index=index_name)

print(es_results['hits']['hits'][0])
# print(result['hits']['hits'][0]['_source']['title'])


titles = []
es_doc_ids = []
for result in es_results['hits']['hits']:
  titles.append(re_norm_nums.sub('N', result["_source"]["title"]))
  es_doc_ids.append(result['_id'])


import nltk
import numpy as np
import orjson
import scipy.spatial.distance as distance
import timeit

from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer

from tunip.corpus_utils_v2 import old_nugget_return_to_v2
from tunip.nugget_api import Nugget, NuggetFilterResultFormat

nugget = Nugget()

start = timeit.default_timer()
white_ptags = ['V', 'N', 'J', 'M', 'SL', 'SH', 'SN']
entries = nugget(titles)

entry_list = list(entries)
# print(entry_list[0])
# print(type(entry_list[0]))
# print(json.dumps(entry_list[0], ensure_ascii=False))
# exit(0)

#corpus_tokens = nugget.filter(nuggets=list(entries), white_tags=white_ptags, result_format=NuggetFilterResultFormat.NUGGET_B_E_LEX)
print(entry_list)
texts = [[t[3] for t in e['tokens']] for e in entry_list]
print(texts)
corpus_tokens = nugget.bigrams_also(texts=texts, white_tags=white_ptags, result_format=NuggetFilterResultFormat.NUGGET_B_E_LEX)
print(f"{timeit.default_timer()-start} for nugget")



# Make bi-gram corpus
start = timeit.default_timer()
corpus = []
features = []
for token_obj in corpus_tokens:
  # tokens = []
  # bigrams = []
  unigrams = [t[2] for t in token_obj]
  # for i in range(0, len(unigrams) - 1):
  #   for j in range(1, len(unigrams)):
  #     if (i == j-1):
  #       bigrams.append('_'.join([unigrams[i], unigrams[j]]))
  # # tokens.extend(unigrams)
  # tokens.extend(bigrams)
  # corpus.append(' '.join(tokens))

  bigrams2 = list(nltk.bigrams(unigrams))
  # print(*map(' '.join, bigrams2), sep=', ')
  corpus.append(' '.join(map('_'.join, bigrams2)))
  # features.append(list(map('_'.join, bigrams2)))

print(f"{timeit.default_timer()-start} for ngram corpus")
print(corpus)



start = timeit.default_timer()
vectorizer = CountVectorizer()
# scipy.sparse._csr.csr_matrix
xx = vectorizer.fit_transform(corpus)
# numpy.ndarray
term_vector = xx.toarray()

print(f"{timeit.default_timer()-start} for vectorizer fit")
# vocab = list(vectorizer.get_feature_names_out())
# print(vocab.index('광역시_최초'))


# TODO from response of hca
docid_clusters = []

for doc_ids in docid_clusters:
  try:
    # cluster_doc_ids = [es_doc_ids[id] for id in doc_ids]
    cluster_doc_ids = []
    for id in doc_ids:
      cluster_doc_ids.append(es_doc_ids[id])
  except IndexError as ie:
    print(id)
    print(len(es_doc_ids))
    exit(0)
  print(cluster_doc_ids)
  cluster_results = list(filter(lambda r: r["_id"] in cluster_doc_ids, es_results['hits']['hits']))

  # print('||\n'.join([corpus[id] for id in doc_ids]))
  print("||\n".join([r["_source"]["title"] for r in cluster_results]))

  print('====')
  

exit(0)


from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.show()




# # make bigram matrix
# data = np.array([list(range(1, n))]).repeat(2, axis=0)
# offsets = np.array([0, -1])
# bigram_mtx = dia_matrix((data, offsets), shape=(n,n)).toarray()

# print(f"{timeit.default_timer()-start} for ngram corpus by scipy dia_matrix")
# print(bigram_mtx)

# # make jaccard between diagonal item and its prior neighbor
# for i, bm_row in enumerate(bigram_mtx):
#   if i > 0 and i < bigram_mtx.shape[0]-2:
#     xx0 = bigram_mtx[i][i-1]
#     xx1 = bigram_mtx[i][i]
#     sims.append(distance.jaccard(xxa[xx0], xxa[xx1]))
