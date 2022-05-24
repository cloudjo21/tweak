import fastcluster
import numpy as np

from collections import deque
from copy import deepcopy
from pydantic import BaseModel
from typing import List, Optional

from tweak.nn_clustering import Linkage


class HCARequest(BaseModel):
    distances: List[np.float64]
    hac_dist_threshold: float
    method: Optional[str] = 'complete'


class HCAResponse(BaseModel):
    # List[List[int]]
    docid_clusters: list


class LinkageLinker:

    def __init__(self, hac_dist_threshold):
        self.hac_dist_threshold = hac_dist_threshold


    def __call__(self, linkage_matrix):
        # Dict[cid, List[Linkage]]
        id2lineage = dict()
        linkages: List[Linkage] = []

        # build_and_cut
        target_num = len(linkage_matrix)
        lk_cid = target_num 
        for lk in linkage_matrix:
            linkages.append(Linkage(id=int(lk[0]), nn_id=int(lk[1]), dist=lk[2], n_clu=int(lk[3]), cid=lk_cid))
            lk_cid += 1
            if lk[2] > self.hac_dist_threshold:
                break

        cid_deq = deque(deepcopy(linkages))
        while len(cid_deq) > 0:
            linkage_item = cid_deq.popleft()
            if linkage_item.n_clu == 2:
                id2lineage[linkage_item.cid] = [linkage_item]
            elif linkage_item.n_clu > 2:
                lineages = id2lineage.get(linkage_item.nn_id) or None
                if lineages:
                    linkage_branch = lineages[-1]

                    id2lineage[linkage_item.cid] = id2lineage[linkage_branch.cid]
                    id2lineage[linkage_item.cid].append(linkage_item)
                    del id2lineage[linkage_branch.cid]


class HCA:
    def __init__(self, hca_dist_thresold):
        self.linkage_linker = LinkageLinker(hca_dist_thresold)

    def __call__(self, request: HCARequest):

        # linkage matrix sorted by ascending distance
        linkage_matrix = fastcluster.linkage(request.distances, method=request.method, preserve_input='True')
        # the number of target documents
        target_num = len(linkage_matrix) + 1

        # build_and_cut
        clu2linkages = self.linkage_linker(linkage_matrix)
        docid_clusters = self._cluster(clu2linkages)

        return HCAResponse(docid_clusters)


    def _cluster(self, clu2linkages: dict, target_num):
        docid_clusters = []
        for k, v in clu2linkages.items():
            doc_ids = [v[0].id, v[0].nn_id]
            if len(v) > 1:
                doc_ids.extend([l.id for l in v[1:] if l.id < target_num])
            docid_clusters.append(doc_ids)

        return docid_clusters
