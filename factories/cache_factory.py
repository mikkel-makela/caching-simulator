import random
from typing import List

from policies.expert_policies.ftpl_policy import ExpertFTPLPolicy
from policies.expert_policies.iawm_policy import IAWMPolicy
from policies.ftpl_policy import FTPLPolicy


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
) -> List[FTPLPolicy]:
    if 1.0 not in discount_rates:
        discount_rates.append(1.0)
    return list(map(lambda d: FTPLPolicy(cache_size, catalog_size, time_horizon, discount_rate=d), discount_rates))


def get_client_cache_connections(clients: int, caches: int, d_regular_degree: int) -> List[List[int]]:
    """
    Gets client cache connections where every cache has d connections.

    :param clients: Number of clients
    :param caches: Number of caches
    :param d_regular_degree: Number of connections per cache
    :return: Client - cache connections
    """
    clients = list(range(clients))
    connections: List[List[int]] = [[] for _ in clients]
    for cache in range(caches):
        for client in random.sample(clients, k=d_regular_degree):
            connections[client].append(cache)
    return connections
