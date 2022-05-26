import numpy as np

from collections import deque
from copy import deepcopy
from typing import List

from tweak.clustering import Linkage


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
