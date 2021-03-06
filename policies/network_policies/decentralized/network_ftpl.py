from typing import List

from policies.ftpl_policy import FTPLPolicy
from policies.network_policies.decentralized.decentralized_policy import DecentralizedNetworkPolicy


class NetworkFTPL(DecentralizedNetworkPolicy):

    policies: List[FTPLPolicy]

    def __init__(
            self,
            cache_count: int,
            client_cache_connections: List[List[int]],
            catalog_size: int,
            cache_size: int,
            time_horizon: int
    ):
        super().__init__(cache_count, client_cache_connections, catalog_size, cache_size)
        self.policies: List[FTPLPolicy] = [
            FTPLPolicy(cache_size, catalog_size, time_horizon, 1.0)
            for _ in range(cache_count)
        ]

    @staticmethod
    def get_name() -> str:
        return "Network FTPL"
