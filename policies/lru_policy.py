from policies.eviction_policy import EvictionPolicy


class LRUPolicy(EvictionPolicy):

    """
    Map that maps an item to a time slot when it was last used.
    """
    _usage_map: dict[int, int]

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._usage_map = dict()

    @staticmethod
    def get_name() -> str:
        return "LRU"

    """
    Learns from the request. Updates the usage map.
    """
    def learn(self, request: int):
        assert request in self.cache
        self._usage_map[request] = self.time

    """
    Removes the item from the cache that has been least recently used.
    """
    def evict_item(self) -> int:
        victim = super().evict_item()
        self._usage_map.pop(victim)
        return victim

    """
    Get the least recently used element.
    """
    def get_victim(self) -> int:
        return min(self._usage_map.items(), key=lambda item: item[1])[0]

    """
    Resets the cache, deleting all entries.
    """
    def reset(self) -> None:
        super().reset()
        self._usage_map = dict()
