from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer

from tunip.nugget_api import Nugget, NuggetFilterResultFormat

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
    text = ''.join(list(filter(lambda c: c.isdigit() or is_hangul(c) or c == ' ', text)))
    text = re.sub('  ', ' ', text)
    return text

max_row = 70
linkage_method = 'average'
# dist_threshold = 0.777
# dist_threshold = 0.822
# dist_threshold = 0.85
dist_threshold = 0.92
# dist_threshold = 0.94
use_bigrams_also = True

white_ptags = {
    'unigram': ['V', 'N', 'SL', 'SH', 'SN'],
    # 'unigram': ['V', 'N', 'J', 'M', 'SL', 'SH', 'SN'],
    'bigram': ['V', 'N', 'J', 'M', 'SL', 'SH', 'SN'],
    #'bigram': ['V', 'N', 'M', 'SL', 'SH', 'SN'],
}

import os
import pathlib

print(str(pathlib.Path(os.curdir).absolute()))

print(os.curdir)
json_path = 'tweak/test/tweak/resources/reserve_keyword_NETFLIX_dashi-nnc_beta6.json'
with open(json_path, mode='r') as io:
    es_result = json.loads(io.read())
# es_result = json.load()

print(json.dumps(es_result, indent=4))


nugget = Nugget()
distance_calc = BigramJaccardDistanceCalc()


titles = [e['_source']['title'] for e in es_result["es_result"]['hits']['hits']]
# titles = [normalize_digit(t) for t in titles]
titles = [preprocess_for_title(t) for t in titles]
es_doc_ids = [e['_id'] for e in es_result["es_result"]['hits']['hits']]
# nuggets = list(nugget(titles))
# title_nuggets = [[t[3] for t in e['tokens']] for e in nuggets]

# print(title_nuggets[0])
# print(title_nuggets[-1])

corpus_bigrams = []
corpus_unigrams = []
bigrams, unigrams = nugget.bigrams_also_selective_tags(texts=titles, white_tags_dict=white_ptags)
# bigrams, unigrams = nugget.bigrams_also(texts=titles, white_tags=white_ptags, result_format=NuggetFilterResultFormat.NUGGET_B_E_LEX)

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

vectorizer = CountVectorizer()

xx = vectorizer.fit_transform(corpus)
# numpy.ndarray
term_vector = xx.toarray()

print(len(term_vector))


dist_calc_req = JaccardDistanceCalcRequest(term_vector= term_vector, num_rows=max_row) # TODO set config from deploy
dist_calc_res: JaccardDistanceCalcResponse = distance_calc(dist_calc_req)

comb_indices = list(combinations(range(0, max_row), 2))
print(len(dist_calc_res.distances))

pidx2dist = dict(list(zip(comb_indices, dist_calc_res.distances)))
    
print(dist_calc_res.distances[0])

hac = NncFactory.create("HAC", dist_threshold)
nnc_req = NncRequest(
    distances=dist_calc_res.distances,
    dist_calc_status=dist_calc_res.status,
    method=linkage_method
)
# docid_clusters = hac(nnc_req)
docid_clusters, dist_clusters = hac.detail(nnc_req)

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
