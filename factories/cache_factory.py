import random
from typing import List

import numpy as np

from system.nodes.cache_node import CacheNode
from system.cache_system import CacheSystem
from system.client import Client
from system.nodes.main_node import MainNode
from policies.expert_policies.ftpl_policy import ExpertFTPLPolicy
from policies.ftrl_policy import FTRLPolicy
from policies.lfu_policy import LFUPolicy
from policies.lru_policy import LRUPolicy
from policies.policy import Policy


def get_expert_ftpl_policy(cache_size: int) -> ExpertFTPLPolicy:
    return ExpertFTPLPolicy(cache_size, [LRUPolicy(cache_size), LFUPolicy(cache_size)])


def get_bipartite_systems_from_datasets(
        datasets: np.ndarray,
        cache_size: int,
        d_regular_degree: int,
        cache_count: int
) -> List[CacheSystem]:
    """
    Gets a bipartite system from datasets. Creates one client node for every dataset, and connects every client
    to d_regular_degree caches.
    :param datasets: 2D numpy array where each subarray represents the trace of some client.
    :param cache_size: The size of every cache in the network.
    :param d_regular_degree: Users per cache.
    :param cache_count: The amount of caches.
    :return: List of caching systems with LRU, LFU, and FTRL caches.
    """
    return _get_bipartite_systems(
        [
            _get_bipartite_lfu_policies(cache_size, cache_count),
            _get_bipartite_lru_policies(cache_size, cache_count),
            _get_bipartite_ftrl_policies(datasets, cache_size, cache_count)
        ],
        d_regular_degree,
        datasets.size
    )


def _get_bipartite_lfu_policies(
        cache_size: int,
        cache_count: int
) -> List[LFUPolicy]:
    assert cache_size > 0
    return [
        LFUPolicy(cache_size)
        for _ in range(cache_count)
    ]


def _get_bipartite_lru_policies(cache_size: int, cache_count: int) -> List[LRUPolicy]:
    assert cache_size > 0
    return [
            LRUPolicy(cache_size)
            for _ in range(cache_count)
    ]


def _get_bipartite_ftrl_policies(
        datasets: np.ndarray,
        cache_size: int,
        cache_count: int
) -> List[FTRLPolicy]:
    assert datasets.size > 0
    assert cache_size > 0
    catalog_size = max(datasets, key=lambda ds: ds.catalog_size).catalog_size
    estimated_trace_size = int(np.average(np.array(list(map(lambda ds: ds.trace.size, datasets)))))
    return [
            FTRLPolicy(cache_size, catalog_size, estimated_trace_size)
            for _ in range(cache_count)
    ]


def _get_bipartite_systems(leaf_policies: List[List[Policy]], d_regular_degree: int, clients: int) -> List[CacheSystem]:
    """
    Gets a list of caching systems.
    :param leaf_policies: List of lists where every inner list is of equal size.
    :param d_regular_degree: Clients per cache.
    :param clients: The amount of clients.
    :return: The list of caching systems.
    """
    leaf_parent_costs: List[int] = [random.randint(1, 100) for _ in range(len(leaf_policies[0]))]
    client_leaf_connections: List[List[int]] = [
        random.sample(list(np.arange(0, len(leaf_policies[0]))), d_regular_degree)
        for _ in range(clients)
    ]
    client_leaf_costs: List[int] = [random.randint(1, 100) for _ in range(clients)]
    return list(
        map(
            lambda policies: _get_bipartite_system(
                policies,
                leaf_parent_costs,
                client_leaf_costs,
                client_leaf_connections
            ),
            leaf_policies
        )
    )


def _get_bipartite_system(
        policies: List[Policy],
        leaf_parent_costs: List[int],
        client_leaf_costs: List[int],
        client_leaf_connections: List[List[int]]
) -> CacheSystem:
    leafs: List[CacheNode] = list(
        map(
            lambda cost_policy_pair: CacheNode(cost_policy_pair[0], cost_policy_pair[1], []),
            list(zip(leaf_parent_costs, policies))
        )
    )
    users: List[Client] = list(
        map(
            lambda connections: Client(list(zip(client_leaf_costs, list(map(lambda i: leafs[i], connections))))),
            client_leaf_connections
        )
    )
    return CacheSystem(MainNode(leafs), users, policies[0].get_name())

