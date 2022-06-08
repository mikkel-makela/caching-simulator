from typing import List

import numpy as np

from policies.network_policies.solvers.lead_cache_solver import get_opt_lead_cache, LeadCacheSolverParams


class LeadCache:
    """
    Regret-optimal caching policy for bi-partite networks with nonlinear reward functions.
    """

    cache_count: int
    client_cache_connections: List[List[int]]
    request_counts: np.ndarray
    time: int
    max_degree: int
    cache_size: int
    configuration: np.ndarray

    def __init__(
            self,
            cache_count: int,
            client_cache_connections: List[List[int]],
            catalog_size: int,
            max_degree: int,
            cache_size: int
    ):
        self.cache_count = cache_count
        self.client_cache_connections = client_cache_connections
        self.request_counts = np.ndarray((len(client_cache_connections), catalog_size))
        self.time = 1
        self.max_degree = max_degree
        self.cache_size = cache_size
        self.configuration = np.zeros((cache_count, catalog_size))

    def update(self, requests: np.ndarray) -> None:
        """
        Updates the cache configuration from new requests.

        :param requests: clients array for requests, where the element at index c is the request by client c
        :return: None
        """
        for client, request in enumerate(requests):
            self.request_counts[client][request] += 1

        learning_rate = (len(self.client_cache_connections) ** 3 / 4) * np.sqrt(
            self.time / (self.cache_size * self.cache_count)
        ) / (
                2 * self.max_degree * (np.log(self.request_counts.shape[1] / self.cache_size) + 1)
        ) ** (1 / 4)

        o_t = self.request_counts + np.random.normal(
            loc=0,
            scale=learning_rate,
            size=self.request_counts.shape
        )

        self.configuration = get_opt_lead_cache(
            LeadCacheSolverParams(
                theta=o_t,
                client_cache_map=self.client_cache_connections,
                catalog_size=self.request_counts.shape[1],
                cache_count=self.cache_count,
                cache_size=self.cache_size
            )
        )

        self.time += 1
