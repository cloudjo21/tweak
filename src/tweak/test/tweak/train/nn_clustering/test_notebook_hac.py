import nltk
import os
import pathlib

from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer

from tunip.nugget_api import Nugget, NuggetFilterResultFormat

from tweak import LOGGER
from tweak.clustering import NncRequest
from tweak.clustering.distance.bigram_jaccard_dist import (
    BigramJaccardDistanceCalc,
    JaccardDistanceCalcRequest,
    JaccardDistanceCalcResponse,
)
from tweak.clustering.factory import NncFactory
from tweak.utils.text_normalize import normalize_digit

import re
import json
from tunip.Hangulpy import is_hangul

regex_normalize_nums = re.compile('\d+')

def preprocess_for_title(text):
    text = regex_normalize_nums.sub('0', text)
    text = ' '.join(map(lambda w: ''.join(filter(lambda c: c.isdigit() or is_hangul(c), w)), text.split(' ')))
    text = re.sub('  ', ' ', text)
    return text


max_row = 70
linkage_method = 'average'
# dist_threshold = 0.777
# dist_threshold = 0.822
dist_threshold = 0.85
# dist_threshold = 0.90
# dist_threshold = 0.92
# dist_threshold = 0.94
use_bigrams_also = True

white_ptags = {
    'unigram': ['V', 'N', 'SL', 'SH', 'SN'],
    'bigram': ['V', 'N', 'J', 'M', 'SL', 'SH', 'SN'],
}


print(str(pathlib.Path(os.curdir).absolute()))

print(os.curdir)
# json_path = 'tweak/test/tweak/resources/reserve_keyword_NETFLIX_dashi-nnc_beta6.json'
# json_path = 'tweak/test/tweak/resources/keyword_오징어_search.json'
json_path = 'tweak/test/tweak/resources/reserve_keyword_AMAZON_ALL_dashi-nnc-beta9.json'
with open(json_path, mode='r') as io:
    es_result = json.loads(io.read())
# es_result = json.load()

# print(json.dumps(es_result, indent=4))


nugget = Nugget()
distance_calc = BigramJaccardDistanceCalc()


corpus_bigrams = []
corpus_unigrams = []
es_doc_ids_wo_nugget = []
titles_wo_nugget = []

es_result_no_titlenugget = [e for e in filter(lambda e: 'titlenugget' not in e['_source'], es_result["es_result"]['hits']['hits'])]
titles_wo_nugget = [e['_source']['title'] for e in filter(lambda e: 'titlenugget' not in e['_source'], es_result["es_result"]['hits']['hits'])]
print(titles_wo_nugget[14])
titles_preprocessed = [preprocess_for_title(t) for t in titles_wo_nugget]
print(titles_preprocessed[14])
es_doc_ids_wo_nugget = [e['_id'] for e in es_result_no_titlenugget]

bigrams, unigrams = nugget.bigrams_also_selective_tags(texts=titles_preprocessed, white_tags_dict=white_ptags)
# bigrams, unigrams = nugget.bigrams_also(texts=titles, white_tags=white_ptags, result_format=NuggetFilterResultFormat.NUGGET_B_E_LEX)
for b in bigrams:
    b = sorted(b)
for u in unigrams:
    u = sorted(u)

titles = [e['_source']['titlenugget'] for e in filter(lambda e: e['_id'] not in es_doc_ids_wo_nugget, es_result['es_result']['hits']['hits'])]
es_doc_ids = [e['_id'] for e in filter(lambda e: e['_id'] not in es_doc_ids_wo_nugget, es_result['es_result']['hits']['hits'])]
# titles = [e['_source']['titlenugget'] for e in es_result["es_result"]['hits']['hits']]
for title in titles:
    title_normed = preprocess_for_title(title).split(' ')
    unigrams.append(sorted(list(set(title_normed))))
    bigrams.append(sorted(nltk.bigrams(title_normed)))


for idx, (id, title) in enumerate(zip(es_doc_ids_wo_nugget, titles_wo_nugget)):
    LOGGER.debug(f"[{idx}]{id}:\t\t\t{title}")
for idx, (id, title) in enumerate(zip(es_doc_ids, titles), len(es_doc_ids_wo_nugget)):
    LOGGER.debug(f"[{idx}]{id}:\t\t\t{title}")


count_unigram = 0
count_bigram = 0
for l in unigrams:
    count_unigram += len(l)
for l in bigrams:
    count_bigram += len(l)

print('count_unigram')
print(count_unigram)
print('count_bigram')
print(count_bigram)

for sent in bigrams:
    corpus_bigrams.append(' '.join(map('_'.join, sent)))

for sent in unigrams:
    corpus_unigrams.append(' '.join(sent))

corpus = []
for us, bs in zip(corpus_unigrams, corpus_bigrams):
    if use_bigrams_also:
        corpus.append(us + ' ' + bs)
    else:
        corpus.append(bs)

# corpus = sorted(corpus)
print('\n'.join(corpus))

vectorizer = CountVectorizer()

xx = vectorizer.fit_transform(corpus)
# numpy.ndarray
term_vector = xx.toarray()

print('len(term_vector):')
print(len(term_vector))
print(term_vector.shape)


dist_calc_req = JaccardDistanceCalcRequest(term_vector= term_vector, num_rows=max_row) # TODO set config from deploy
dist_calc_res: JaccardDistanceCalcResponse = distance_calc(dist_calc_req)

comb_indices = list(combinations(range(0, max_row), 2))
print(len(dist_calc_res.distances))

pidx2dist = dict(list(zip(comb_indices, dist_calc_res.distances)))
    

hac = NncFactory.create("HAC", dist_threshold)
nnc_req = NncRequest(
    distances=dist_calc_res.distances,
    dist_calc_status=dist_calc_res.status,
    method=linkage_method
)
LOGGER.debug(dist_calc_res.distances)
# exit(0)
# docid_clusters = hac(nnc_req)
docid_clusters, dist_clusters = hac.detail(nnc_req)

print('docid_clusters:')
print(docid_clusters)
print(dist_clusters)


print('================================')

# print('\n'.join([corpus[28], corpus[30], corpus[40]]))

num_clus = []
for cids, dists in zip(docid_clusters, dist_clusters):
    print(' | '.join(list(map(lambda d: str(d), dists))))
    print('\n'.join(list(map(lambda c: f"cid:{c}: {corpus[c]}", cids))))
    print('================================')
    num_clus.append(len(cids))

avg_items_clu = sum(num_clus) / len(docid_clusters)

print(f"linkage method: {linkage_method}")
print(f"pair distance threshold for clustering: {dist_threshold}")
print(f"use bigrams and unigrams: {use_bigrams_also}")
print(f"{len(docid_clusters)} number of clusters")
print(f"{avg_items_clu} average number of items for each cluster")
