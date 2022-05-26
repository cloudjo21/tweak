import fastcluster
import numpy as np

from copy import deepcopy
from pydantic import BaseModel
from typing import List, Optional

from tweak.clustering.linkage import LinkageLinker


class HAC:
    """
    Hierarchical Agglomerative Clustering
    """
    def __init__(self, dist_thresold):
        self.linkage_linker = LinkageLinker(dist_thresold)

    def __call__(self, distances: List[np.float64], method: Optional[str]='complete') -> list:

        # linkage matrix sorted by ascending distance
        linkage_matrix = fastcluster.linkage(distances, method=method, preserve_input='True')
        # the number of target documents
        target_num = len(linkage_matrix) + 1

        # build_and_cut
        clu2linkages = self.linkage_linker(linkage_matrix)
        docid_clusters = self._cluster(clu2linkages)

        return docid_clusters


    def _cluster(self, clu2linkages: dict, target_num):
        docid_clusters = []
        for k, v in clu2linkages.items():
            doc_ids = [v[0].id, v[0].nn_id]
            if len(v) > 1:
                doc_ids.extend([l.id for l in v[1:] if l.id < target_num])
            docid_clusters.append(doc_ids)

        return docid_clusters
