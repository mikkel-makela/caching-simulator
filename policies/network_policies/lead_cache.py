from typing import List

import numpy as np

from policies.network_policies.network_policy import NetworkPolicy
from policies.network_policies.solvers.lead_cache_solver import get_opt_lead_cache, LeadCacheSolverParams


class LeadCache(NetworkPolicy):
    """
    Regret-optimal caching policy for bi-partite networks with non-linear reward functions.
    """

    request_counts: np.ndarray
    max_degree: int

    def __init__(
            self,
            cache_count: int,
            client_cache_connections: List[List[int]],
            catalog_size: int,
            max_degree: int,
            cache_size: int
    ):
        super().__init__(cache_count, client_cache_connections, catalog_size, cache_size)
        self.request_counts = np.ndarray((len(client_cache_connections), catalog_size))
        self.max_degree = max_degree

    @staticmethod
    def get_name() -> str:
        return "LeadCache"

    def update(self, requests: np.ndarray) -> None:
        super().update(requests)
        self._update_counts(requests)

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

    def _update_counts(self, requests: np.ndarray) -> None:
        for client, request in enumerate(requests):
            self.request_counts[client][round(request)] += 1
