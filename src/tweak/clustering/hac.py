import fastcluster
import numpy as np

from typing import List, Optional

from tweak.clustering import NncRequest
from tweak.clustering.distance import DistanceCalcStatus, NotSupportedDistanceCalcStatus
from tweak.clustering.linkage.linker import LinkageLinker


class _HAC:
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
        docid_clusters = self._cluster(clu2linkages, target_num)

        return docid_clusters

    def debug(self, distances: List[np.float64], method: Optional[str]='complete') -> list:
        # linkage matrix sorted by ascending distance
        linkage_matrix = fastcluster.linkage(distances, method=method, preserve_input='True')
        # the number of target documents
        target_num = len(linkage_matrix) + 1

        # build_and_cut
        clu2linkages = self.linkage_linker(linkage_matrix)
        docid_clusters, dist_clusters = self._cluster_detail(clu2linkages, target_num)

        return docid_clusters, dist_clusters

    def _cluster(self, clu2linkages: dict, target_num: int):
        docid_clusters = []
        for k, v in clu2linkages.items():
            doc_ids = [v[0].id, v[0].nn_id]
            if len(v) > 1:
                doc_ids.extend([l.id for l in v[1:] if l.id < target_num])
                doc_ids.extend([l.nn_id for l in v[1:] if l.nn_id < target_num])
            docid_clusters.append(doc_ids)

        return docid_clusters

    def _cluster_detail(self, clu2linkages: dict, target_num: int):
        docid_clusters = []
        dist_clusters = []
        for k, v in clu2linkages.items():
            doc_ids = [v[0].id, v[0].nn_id]
            dists = [v[0].dist]
            if len(v) > 1:
                doc_ids.extend([l.id for l in v[1:] if l.id < target_num])
                dists.extend([l.dist for l in v[1:]if l.id < target_num])
                doc_ids.extend([l.nn_id for l in v[1:] if l.nn_id < target_num])
                dists.extend([l.dist for l in v[1:]if l.nn_id < target_num])
            docid_clusters.append(doc_ids)
            dist_clusters.append(dists)

        return docid_clusters, dist_clusters


class HAC:
    def __init__(self, dist_threshold):
        self._model = _HAC(dist_threshold)

    def __call__(self, nnc_request: NncRequest) -> list:
        if nnc_request.dist_calc_status is DistanceCalcStatus.OK:
            return self._model(nnc_request.distances, nnc_request.method)
        elif nnc_request.dist_calc_status is DistanceCalcStatus.ONLY:
            return [[1]]
        elif nnc_request.dist_calc_status is DistanceCalcStatus.EMPTY:
            return [[]]
        else:
            raise NotSupportedDistanceCalcStatus()

    def detail(self, nnc_request: NncRequest) -> tuple:
        if nnc_request.dist_calc_status is DistanceCalcStatus.OK:
            return self._model.debug(nnc_request.distances, nnc_request.method)
        elif nnc_request.dist_calc_status is DistanceCalcStatus.ONLY:
            return [[1]]
        elif nnc_request.dist_calc_status is DistanceCalcStatus.EMPTY:
            return [[]]
        else:
            raise NotSupportedDistanceCalcStatus()
