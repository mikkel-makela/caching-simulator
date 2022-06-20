from typing import List

from policies.lfu_policy import LFUPolicy
from policies.network_policies.decentralized.decentralized_policy import DecentralizedNetworkPolicy


class NetworkLFU(DecentralizedNetworkPolicy):

    def __init__(
            self,
            cache_count: int,
            client_cache_connections: List[List[int]],
            catalog_size: int,
            cache_size: int
    ):
        super().__init__(cache_count, client_cache_connections, catalog_size, cache_size)
        self.policies = [LFUPolicy(cache_size) for _ in range(cache_count)]

    @staticmethod
    def get_name() -> str:
        return "Network LFU"
