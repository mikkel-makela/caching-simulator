from copy import deepcopy

from policies.eviction_policy import EvictionPolicy


class ExpertAdaptedPolicy:
    """
    Uses two caches to return eviction advice.

    Mirror cache - a cache that is kept equivalent to the policy using this expert. This cache must be changed
    via proxy methods for adding and removing items, because calling the main serve_request method would evict a
    custom item.

    Virtual cache - a cache that operates normally like the policy it belongs should. Used to measure the loss of
    the expert.
    """

    mirror_policy: EvictionPolicy
    virtual_policy: EvictionPolicy
    loss: float

    def __init__(self, policy: EvictionPolicy, initial_loss: float):
        self.mirror_policy = policy
        self.virtual_policy = deepcopy(policy)
        self.loss = initial_loss

    def get_eviction_advice(self) -> int:
        return self.mirror_policy.get_victim()

    def evict_item_from_mirror_cache(self, item: int) -> None:
        self.mirror_policy.remove_item_from_cache(item)

    def can_virtual_cache_serve_request(self, item: int) -> bool:
        return self.virtual_policy.is_present(item)

    def add_item_to_mirror_cache(self, item: int) -> None:
        self.mirror_policy.add_item(item)

    def learn_from_request(self, item: int) -> None:
        self.virtual_policy.serve_request(item)
        # Mirror policy does not always have the item it is supposed to learn from.
        self.mirror_policy.learn(item)

    def reset(self) -> None:
        self.virtual_policy.reset()
        self.mirror_policy.reset()
        self.loss = 0

