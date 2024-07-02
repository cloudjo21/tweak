from collections import deque
from copy import deepcopy
from typing import List

from tweak import LOGGER
from tweak.clustering.linkage import Linkage


class LinkageLinker:

    def __init__(self, dist_threshold):
        self.hac_dist_threshold = dist_threshold


    def __call__(self, linkage_matrix):
        # Dict[cid, List[Linkage]]
        id2lineage = dict()
        linkages: List[Linkage] = []

        # build_and_cut
        target_num = len(linkage_matrix) + 1
        lk_cid = target_num 
        for lk in linkage_matrix:
            if lk[2] < self.hac_dist_threshold:
                linkages.append(Linkage(id=int(lk[0]), nn_id=int(lk[1]), dist=lk[2], n_clu=int(lk[3]), cid=lk_cid))
            else:
                break
            lk_cid += 1
        
        LOGGER.debug('\n'.join(list(map(lambda l: f"id={l.id}, nn_id={l.nn_id}, dist={l.dist}, cid={l.cid} n_clu={l.n_clu}", linkages))))

        cid_deq = deque(deepcopy(linkages))
        while len(cid_deq) > 0:
            linkage_item = cid_deq.popleft()
            if linkage_item.n_clu == 2:
                id2lineage[linkage_item.cid] = [linkage_item]
            elif linkage_item.n_clu > 2:
                id2lineage[linkage_item.cid] = []

                lineages = id2lineage.get(linkage_item.nn_id) or None
                if lineages:
                    id2lineage[linkage_item.cid].extend(lineages)
                    del id2lineage[linkage_item.nn_id]
                else:
                    id2lineage[linkage_item.cid].append(linkage_item)

                lineages = id2lineage.get(linkage_item.id) or None
                if lineages:
                    id2lineage[linkage_item.cid].extend(lineages)
                    del id2lineage[linkage_item.id]
                else:
                    id2lineage[linkage_item.cid].append(linkage_item)
        
        LOGGER.debug('\n'.join(map(lambda p: f"{p[0]}: {p[1]}", id2lineage.items())))

        return id2lineage
