from copy import deepcopy

from policies.policy import Policy


class ExpertAdaptedPolicy:

    mirror_policy: Policy
    virtual_policy: Policy
    loss: float

    def __init__(self, policy: Policy, initial_loss: float):
        self.mirror_policy = policy
        self.virtual_policy = deepcopy(policy)
        self.loss = initial_loss

    def get_eviction_advice(self) -> int:
        return self.mirror_policy.get_victim()

    def evict_item_from_mirror_cache(self, item: int) -> None:
        self.mirror_policy.remove_item_from_cache(item)

    def process_request(self, item: int) -> None:
        self._update_mirroring_cache(item)
        self._update_virtual_cache(item)

    def can_virtual_cache_serve_request(self, item: int) -> bool:
        return self.virtual_policy.is_present(item)

    def _update_mirroring_cache(self, item: int) -> None:
        self.mirror_policy.add_item(item)

    def _update_virtual_cache(self, item: int) -> None:
        self.virtual_policy.serve_request(item)
