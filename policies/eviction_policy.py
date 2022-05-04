import numpy as np

from policies.policy import Policy


class EvictionPolicy(Policy):
    """
    Abstract class for policies that make single evictions when their caches become full.
    """

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
            self.add_file(request)

        self.learn(request)

    """
    Evicts an item from the cache.
    """
    def evict_item(self) -> int:
        victim = self.get_victim()
        return self.remove_file_from_cache(victim)

    """
    Removes the item from the cache array.
    """
    def remove_file_from_cache(self, file: int) -> int:
        self.cache[np.where(self.cache == file)] = None
        return file

    """
    Adds an item to a cache, assumes that the cache is not full.
    """
    def add_file(self, file: int) -> None:
        assert not self.is_full()
        self.cache[np.where(self.cache == None)[0][0]] = file

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


