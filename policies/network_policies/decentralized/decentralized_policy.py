import random
from typing import List, Dict

import numpy as np

from policies.network_policies.network_policy import NetworkPolicy
from policies.policy import Policy


class DecentralizedNetworkPolicy(NetworkPolicy):

    policies: List[Policy]

    def update(self, requests: np.ndarray) -> None:
        """
        Updates the cache configuration from new requests.

        :param requests: clients array for requests, where the element at index c is the request by client c
        :return: None
        """
        super().update(requests)
        cache_requests_map: Dict[int, set[int]] = dict()
        for client, request in enumerate(requests):
            for cache in self.client_cache_connections[client]:
                if cache not in cache_requests_map:
                    cache_requests_map[cache] = set()
                cache_requests_map[cache].add(int(request))

        for cache, requests in cache_requests_map.items():
            requests = list(requests)
            random.shuffle(requests)
            for request in requests:
                self.policies[cache].update(request)

        self.configuration = np.zeros(self.configuration.shape)
        for cache, policy in enumerate(self.policies):
            for file in policy.cache:
                if file is not None:
                    self.configuration[cache][file] = 1



