from typing import List

import numpy as np

from policies.expert_adapter.expert_adapter import ExpertAdaptedPolicy
from policies.policy import Policy


class ExpertPolicy(Policy):

    experts: List[ExpertAdaptedPolicy]

    def __init__(self, capacity: int, policies: List[Policy], initial_losses: np.ndarray = None):
        super().__init__(capacity)
        initial_losses = np.zeros(len(policies)) if initial_losses is None else initial_losses
        self.experts = list(map(lambda pair: ExpertAdaptedPolicy(pair[0], pair[1]), zip(policies, initial_losses)))

    """
    Updates the cache with the new request.
    """
    def serve_request(self, request: int) -> None:
        self.record_losses(request)
        super().serve_request(request)

    """
    Learns from the request. Updates the experts.
    """
    def learn(self, request: int):
        for expert in self.experts:
            expert.learn_from_request(request)

    """
    Evicts item from own and all expert mirroring caches. Records the eviction to compute losses.
    """
    def evict_item(self) -> None:
        victim_item = super().evict_item()
        for expert in self.experts:
            expert.evict_item_from_mirror_cache(victim_item)

    """
    Adds an item to own and all expert caches.
    """
    def add_item(self, item: int) -> None:
        assert not self.is_full()
        super().add_item(item)
        for expert in self.experts:
            expert.add_item_to_mirroring_cache(item)

    """
    Records losses based on whether the request can be served.
    """
    def record_losses(self, request: int) -> None:
        pass
