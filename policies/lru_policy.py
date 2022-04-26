from policies.policy import Policy


class LRUPolicy(Policy):

    """
    Map that maps an item to a time slot when it was last used.
    """
    _usage_map: dict[int, int]

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._usage_map = dict()

    @staticmethod
    def get_name() -> str:
        return "LRU Policy"

    """
    Learns from the request. Updates the usage map.
    """
    def learn(self, request: int):
        assert request in self.cache
        self._usage_map[request] = self.time

    """
    Adds an item to a cache, assumes that the cache is not full.
    """
    def add_item(self, item: int) -> None:
        assert item not in self.cache
        super().add_item(item)
        self._usage_map[item] = self.time

    """
    Removes the item from the cache that has been least recently used.
    """
    def remove_item_from_cache(self, item: int) -> int:
        self._usage_map.pop(item)
        return super().remove_item_from_cache(item)

    """
    Get the least recently used element.
    """
    def get_victim(self) -> int:
        return max(self._usage_map.items(), key=lambda item: self.time - item[1])[0]

    """
    Resets the cache, deleting all entries.
    """
    def reset(self) -> None:
        super().reset()
        self._usage_map = dict()
