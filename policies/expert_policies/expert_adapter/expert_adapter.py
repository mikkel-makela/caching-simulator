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

    def evict_file_from_mirror_cache(self, file: int) -> None:
        self.mirror_policy.remove_file_from_cache(file)

    def can_virtual_cache_serve_request(self, file: int) -> bool:
        return self.virtual_policy.is_present(file)

    def add_file_to_mirror_cache(self, file: int) -> None:
        self.mirror_policy.add_file(file)

    def learn_from_request(self, file: int) -> None:
        self.virtual_policy.update(file)
        # Mirror policy does not always have the file it is supposed to learn from.
        self.mirror_policy.learn(file)

    def reset(self) -> None:
        self.virtual_policy.reset()
        self.mirror_policy.reset()
        self.loss = 0

