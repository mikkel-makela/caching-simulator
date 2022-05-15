from policies.eviction_policy import EvictionPolicy


class LFUPolicy(EvictionPolicy):

    """
    Map that maps a file to a how many times it has been used.
    """
    _frequency_map: dict[int, int]

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._frequency_map = dict()

    @staticmethod
    def get_name() -> str:
        return "LFU"

    """
    Learns from the request. Updates the frequency map.
    """
    def learn(self, request: int):
        if self.is_present(request):
            self._frequency_map[request] += 1

    """
    Adds an item to a cache, assumes that the cache is not full.
    """
    def add_file(self, file: int) -> None:
        super().add_file(file)
        self._frequency_map[file] = 0

    """
    Evicts an item from the cache.
    """
    def evict_item(self) -> int:
        victim = super().evict_item()
        self._frequency_map.pop(victim)
        return victim


    """
    Resets the cache, deleting all entries.
    """
    def reset(self) -> None:
        super().reset()
        self._frequency_map = dict()

    """
    Get the least frequently used element.
    """
    def get_victim(self) -> int:
        return min(self._frequency_map.items(), key=lambda item: item[1])[0]
