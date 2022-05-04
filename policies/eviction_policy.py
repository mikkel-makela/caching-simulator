import numpy as np

from policies.policy import Policy


class EvictionPolicy(Policy):

    def __init__(self, capacity: int):
        super().__init__(capacity)

    """
    Updates the cache with the new request.
    """
    def update(self, request: int) -> None:
        super().update(request)
        if not self.is_present(request):
            if self.is_full():
                self.evict_item()

            assert not self.is_full()
            self.add_item(request)

        self.learn(request)

    """
    Evicts an item from the cache.
    """
    def evict_item(self) -> int:
        victim = self.get_victim()
        return self.remove_item_from_cache(victim)

    """
    Removes the item from the cache array.
    """
    def remove_item_from_cache(self, item: int) -> int:
        self.cache[np.where(self.cache == item)] = None
        return item

    """
    Adds an item to a cache, assumes that the cache is not full.
    """
    def add_item(self, item: int) -> None:
        assert not self.is_full()
        self.cache[np.where(self.cache == None)[0][0]] = item

    """
    Get the item to be popped.
    """
    def get_victim(self) -> int:
        pass

    """
    Makes changes to its model from the new request.
    """
    def learn(self, request: int) -> None:
        pass


