from policies.policy import Policy


class LRUPolicy(Policy):

    """
    Map that maps an item to a time slot when it was last used.
    """
    _usage_map: dict[int, int]

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._usage_map = dict()

    """
    Updates the cache with the new request.
    """
    def serve_request(self, request: int) -> None:
        if self.is_present(request):
            self._usage_map[request] += self.time

        super().serve_request(request)

    @staticmethod
    def get_name() -> str:
        return "LRU Policy"

    """
    Adds an item to a cache, assumes that the cache is not full.
    """
    def add_item(self, item: int) -> None:
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
