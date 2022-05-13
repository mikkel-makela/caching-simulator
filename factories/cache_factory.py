import random
from typing import List

import numpy as np

from nodes.cache_node import CacheNode
from nodes.cache_system import CacheSystem
from nodes.client_node import ClientNode
from nodes.main_node import MainNode
from policies.expert_policies.ftpl_policy import ExpertFTPLPolicy
from policies.ftrl_policy import FTRLPolicy
from policies.lfu_policy import LFUPolicy
from policies.lru_policy import LRUPolicy
from policies.policy import Policy


def get_expert_ftpl_policy(cache_size: int) -> ExpertFTPLPolicy:
    return ExpertFTPLPolicy(cache_size, [LRUPolicy(cache_size), LFUPolicy(cache_size)])


def get_bipartite_system_from_dataset(
        datasets: np.ndarray,
        cache_size: int,
        d_regular_degree: int = 1
) -> CacheSystem:
    assert datasets.size > 0
    assert cache_size > 0
    catalog_size = max(datasets, key=lambda ds: ds.catalog_size).catalog_size
    estimated_trace_size = int(np.average(np.array(list(map(lambda ds: ds.trace.size, datasets)))))
    return get_bipartite_system(
        [
            FTRLPolicy(cache_size, catalog_size, estimated_trace_size)
            for _ in range(int(datasets.size / d_regular_degree))
        ],
        d_regular_degree,
        datasets.size
    )


def get_bipartite_system(leaf_policies: List[Policy], d_regular_degree: int, clients: int) -> CacheSystem:
    leafs = [
        CacheNode(random.randint(1, 100), policy, [])
        for policy in leaf_policies
    ]
    users: List[ClientNode] = [
        ClientNode([
            (random.randint(1, 100), cache) for cache in random.sample(leafs, d_regular_degree)
        ]) for _ in range(clients)
    ]
    return CacheSystem(MainNode(leafs), users)
