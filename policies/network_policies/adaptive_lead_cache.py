from typing import List

import numpy as np

from policies.network_policies.decentralized.network_adaptive_ftpl import NetworkAdaptiveFTPL
from policies.network_policies.lead_cache import LeadCache


class AdaptiveLeadCache(LeadCache):

    network_ftpl: NetworkAdaptiveFTPL

    def __init__(
            self,
            cache_count: int,
            client_cache_connections: List[List[int]],
            catalog_size: int,
            max_degree: int,
            cache_size: int,
            discount_rates: List[float],
            time_horizon: int
    ):
        super().__init__(cache_count, client_cache_connections, catalog_size, max_degree, cache_size)
        self.network_ftpl = NetworkAdaptiveFTPL(
            cache_count,
            client_cache_connections,
            catalog_size,
            cache_size,
            discount_rates,
            time_horizon
        )

    @staticmethod
    def get_name() -> str:
        return "Adaptive LeadCache"

    def update(self, requests: np.ndarray) -> None:
        super().update(requests)
        for cache in range(self.cache_count):
            self.network_ftpl.policies[cache].cache = self.configuration[cache]

    def _update_counts(self, requests: np.ndarray) -> None:
        self.network_ftpl.update(requests)
        for cache in range(self.cache_count):
            self.request_counts[cache] = self.network_ftpl.policies[cache].current_expert.policy.file_request_counts

