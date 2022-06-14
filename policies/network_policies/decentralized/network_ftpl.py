from typing import List

from factories.cache_factory import get_expert_iawm_policy
from policies.expert_policies.iawm_policy import IAWMPolicy
from policies.network_policies.decentralized.decentralized_policy import DecentralizedNetworkPolicy


class NetworkFTPL(DecentralizedNetworkPolicy):

    policies: List[IAWMPolicy]

    def __init__(
            self,
            cache_count: int,
            client_cache_connections: List[List[int]],
            catalog_size: int,
            cache_size: int,
            discount_rates: List[float],
            time_horizon: int
    ):
        super().__init__(cache_count, client_cache_connections, catalog_size, cache_size)
        self.policies: List[IAWMPolicy] = [
            get_expert_iawm_policy(cache_size, catalog_size, time_horizon, discount_rates)
            for _ in range(cache_count)
        ]

    @staticmethod
    def get_name() -> str:
        return "Network FTPL"
