from typing import List

from policies.expert_adapter.expert_adapter import ExpertAdaptedPolicy
from policies.policy import Policy


class ExpertPolicy(Policy):

    experts: List[ExpertAdaptedPolicy]

    def __init__(self, capacity: int, policies: [Policy], initial_loss: float = 0):
        super().__init__(capacity)
        self.experts = list(map(lambda policy: ExpertAdaptedPolicy(policy, initial_loss), policies))

    """
    Updates losses, serves the request.
    """
    def serve_request(self, request: int) -> None:
        self.record_losses(request)
        super().serve_request(request)

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
            expert.process_request(item)

    """
    Records losses based on whether the request can be served.
    """
    def record_losses(self, request: int) -> None:
        pass
