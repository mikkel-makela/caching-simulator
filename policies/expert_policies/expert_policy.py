from typing import List

import numpy as np

from policies.expert_policies.expert_adapter.expert_adapter import ExpertAdaptedPolicy
from policies.eviction_policy import EvictionPolicy


class ExpertPolicy(EvictionPolicy):
    """
    Keeps a list of policies that have virtual caches which serve requests normally, and mirror caches which
    are adapted to serve requests such that cache contents are equal to the expert policy.
    """

    experts: List[ExpertAdaptedPolicy]

    def __init__(self, capacity: int, policies: List[EvictionPolicy]):
        super().__init__(capacity)
        initial_losses = np.zeros(len(policies))
        self.experts = list(map(lambda pair: ExpertAdaptedPolicy(pair[0], pair[1]), zip(policies, initial_losses)))

    """
    Updates the cache with the new request.
    """
    def update(self, request: int) -> None:
        self.record_losses(request)
        super().update(request)

    """
    Learns from the request. Updates the experts.
    """
    def learn(self, request: int):
        for expert in self.experts:
            expert.learn_from_request(request)

    """
    Removes item from own and all expert mirroring caches.
    """
    def remove_file_from_cache(self, file: int) -> int:
        for expert in self.experts:
            expert.evict_file_from_mirror_cache(file)
        return super().remove_file_from_cache(file)

    """
    Adds an item to own and all expert caches.
    """
    def add_file(self, file: int) -> None:
        assert not self.is_full()
        super().add_file(file)
        for expert in self.experts:
            expert.add_file_to_mirror_cache(file)

    """
    Records losses based on whether the request can be served.
    """
    def record_losses(self, request: int) -> None:
        pass
