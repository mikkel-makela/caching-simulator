import numpy as np


class Policy:

    """
    Cache of objects. None elements represent free slots.
    """
    cache: np.ndarray

    """
    Current time.
    """
    time: int

    def __init__(self, capacity: int):
        self.time = 1
        self.cache = np.empty(capacity, dtype=object)

    @staticmethod
    def get_name() -> str:
        pass

    """
    Updates the cache with the new request.
    """
    def serve_request(self, request: int) -> None:
        if not self.is_present(request):
            if self.is_full():
                self.evict_item()

            assert not self.is_full()
            self.add_item(request)

        self.learn(request)
        self._advance_time()

    """
    Resets the cache, deleting all entries.
    """
    def reset(self) -> None:
        self.cache = np.empty(len(self.cache), dtype=object)
        self.time = 1

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
    Checks if an item is present in the cache.
    """
    def is_present(self, item: int) -> bool:
        return item in self.cache

    """
    Checks if the cache is full, i.e if it has any None elements which represent free slots.
    """
    def is_full(self) -> bool:
        return None not in self.cache

    """
    Advances time.
    """
    def _advance_time(self) -> None:
        self.time += 1

    """
    Makes changes to its model from the new request.
    """
    def learn(self, request: int) -> None:
        pass

    """
    Get the item to be popped.
    """
    def get_victim(self) -> int:
        pass

