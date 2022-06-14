from typing import List

import numpy as np


class NetworkPolicy:

    cache_count: int
    client_cache_connections: List[List[int]]
    time: int
    cache_size: int
    configuration: np.ndarray
    reward: int

    def __init__(
            self,
            cache_count: int,
            client_cache_connections: List[List[int]],
            catalog_size: int,
            cache_size: int
    ):
        self.cache_count = cache_count
        self.client_cache_connections = client_cache_connections
        self.cache_size = cache_size
        self.configuration = np.zeros((cache_count, catalog_size + 1))
        self.time = 0
        self.reward = 0

    @staticmethod
    def get_name() -> str:
        pass

    def update(self, requests: np.ndarray) -> None:
        """
        Updates the cache configuration from new requests.

        :param requests: clients array for requests, where the element at index c is the request by client c
        :return: None
        """
        self._update_reward(requests)
        self.time += 1

    def _update_reward(self, requests: np.ndarray) -> None:
        for client, request in enumerate(requests):
            if any(
                    map(
                        lambda connection: round(self.configuration[connection][round(request)]) == 1,
                        self.client_cache_connections[client]
                    )
            ):
                self.reward += 1
