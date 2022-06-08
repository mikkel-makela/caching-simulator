import random
from typing import List, Callable

import numpy as np

from policies.expert_policies.ftpl_policy import ExpertFTPLPolicy
from policies.expert_policies.iawm_policy import IAWMPolicy
from system.nodes.cache_node import CacheNode
from system.cache_system import CacheSystem
from system.client import Client
from system.nodes.main_node import MainNode
from policies.ftpl_policy import FTPLPolicy
from policies.lfu_policy import LFUPolicy
from policies.lru_policy import LRUPolicy
from policies.policy import Policy
from system.nodes.node import Node


def get_expert_ftpl_policy(
        cache_size: int,
        catalog_size: int,
        time_horizon: int,
        discount_rates: List[float]
) -> ExpertFTPLPolicy:
    return ExpertFTPLPolicy(
        cache_size,
        get_ftpl_policies(cache_size, catalog_size, time_horizon, discount_rates)
    )


def get_expert_iawm_policy(
        cache_size: int,
        catalog_size: int,
        time_horizon: int,
        discount_rates: List[float]
) -> IAWMPolicy:
    return IAWMPolicy(
        cache_size,
        get_ftpl_policies(cache_size, catalog_size, time_horizon, discount_rates)
    )


def get_ftpl_policies(
        cache_size: int,
        catalog_size: int,
        time_horizon: int,
        discount_rates: List[float]
) -> List[Policy]:
    if 1.0 not in discount_rates:
        discount_rates.append(1.0)
    return list(map(lambda d: FTPLPolicy(cache_size, catalog_size, time_horizon, discount_rate=d), discount_rates))


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
            _get_lfu_policies(cache_size, cache_count),
            _get_lru_policies(cache_size, cache_count),
            _get_ftpl_policies(datasets, cache_size, cache_count),
            _get_iawm_policies(datasets, cache_size, cache_count)
        ],
        d_regular_degree,
        datasets.size
    )


def get_hierarchical_system_from_datasets(
        datasets: np.ndarray,
        cache_size: int,
        cache_size_factor: int,
        d_regular_degree: int,
        layers: int,
        policy_generator: Callable[[int, int], List[Policy]]
) -> CacheSystem:
    """
    Returns a hierarchical caching system. Clients are not connected to the main node, and must acquire all files
    by interfacing with visible caches.

    :param datasets: Datasets of connected users
    :param cache_size: Size of edge caches
    :param cache_size_factor: Factor by which cache size increases as we go up in the hierarchy
    :param d_regular_degree: How many edge caches each user is connected to
    :param layers: The amount of layers in the hierarchy
    :param policy_generator: A generator function that takes in the cache size and the amount of policies to generate.
    :return: Caching system
    """
    def get_edge_nodes() -> int:
        total_nodes = 1
        for i in range(1, layers):
            total_nodes += cache_size_factor ** i
        return total_nodes

    assert layers >= 1
    edge_nodes = get_edge_nodes()
    assert datasets.size >= 2 * edge_nodes

    main_node = MainNode()

    return CacheSystem([], "Insert policy name")


def _get_lfu_policies(
        cache_size: int,
        cache_count: int
) -> List[LFUPolicy]:
    assert cache_size > 0
    return [
        LFUPolicy(cache_size)
        for _ in range(cache_count)
    ]


def _get_lru_policies(cache_size: int, cache_count: int) -> List[LRUPolicy]:
    assert cache_size > 0
    return [
            LRUPolicy(cache_size)
            for _ in range(cache_count)
    ]


def _get_ftpl_policies(
        datasets: np.ndarray,
        cache_size: int,
        cache_count: int
) -> List[FTPLPolicy]:
    catalog_size, estimated_trace_size = _get_catalog_and_trace_size(datasets, cache_size)
    return [
            FTPLPolicy(cache_size, catalog_size, estimated_trace_size)
            for _ in range(cache_count)
    ]


def _get_iawm_policies(
        datasets: np.ndarray,
        cache_size: int,
        cache_count: int
) -> List[IAWMPolicy]:
    catalog_size, estimated_trace_size = _get_catalog_and_trace_size(datasets, cache_size)
    return [
            get_expert_iawm_policy(cache_size, catalog_size, estimated_trace_size, [0.99, 0.999, 0.9999, 1.0])
            for _ in range(cache_count)
    ]


def _get_catalog_and_trace_size(datasets: np.ndarray, cache_size: int) -> tuple[int, int]:
    assert datasets.size > 0
    assert cache_size > 0
    catalog_size = max(datasets, key=lambda ds: ds.catalog_size).catalog_size
    estimated_trace_size = int(np.average(np.array(list(map(lambda ds: ds.trace.size, datasets)))))
    return catalog_size, estimated_trace_size


def _get_estimated_bipartite_time_horizon(datasets: np.ndarray) -> int:
    return int(np.average(np.array(list(map(lambda ds: ds.trace.size, datasets)))))


def _get_bipartite_catalog_size(datasets: np.ndarray) -> int:
    return max(datasets, key=lambda ds: ds.catalog_size).catalog_size


def _get_bipartite_systems(leaf_policies: List[List[Policy]], d_regular_degree: int, clients: int) -> List[CacheSystem]:
    """
    Gets a list of caching systems.
    :param leaf_policies: List of lists where every inner list is of equal size.
    :param d_regular_degree: Clients per cache.
    :param clients: The amount of clients.
    :return: The list of caching systems.
    """
    leaf_parent_costs: List[float] = [random.random() * 0.1 for _ in range(len(leaf_policies[0]))]
    client_cache_connections: List[List[int]] = [
        random.sample(list(np.arange(0, len(leaf_policies[0]))), d_regular_degree)
        for _ in range(clients)
    ]
    return list(
        map(
            lambda policies: _get_bipartite_system(
                policies,
                leaf_parent_costs,
                client_cache_connections
            ),
            leaf_policies
        )
    )


def _get_bipartite_system(
        policies: List[Policy],
        leaf_parent_costs: List[float],
        client_cache_connections: List[List[int]]
) -> CacheSystem:
    leafs: List[Node] = list(
        map(
            lambda policy: CacheNode(policy),
            policies
        )
    )
    main_node: Node = MainNode()
    dummy_nodes: List[CacheNode] = list(
        map(
            lambda connections: CacheNode(
                LRUPolicy(0),
                list(zip(leaf_parent_costs, list(map(lambda i: leafs[i], connections))))
                + [(10000.0, main_node)]
            ),
            client_cache_connections
        )
    )
    clients: List[Client] = list(map(lambda node: Client(node, 0), dummy_nodes))
    return CacheSystem(clients, policies[0].get_name(), leafs)

