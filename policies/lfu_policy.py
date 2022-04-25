from policies.policy import Policy


class LFUPolicy(Policy):

    """
    Map that maps an item to a how many times it has been used.
    """
    _frequency_map: dict[int, int]

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._frequency_map = dict()

    @staticmethod
    def get_name() -> str:
        return "LFU Policy"

    """
    Updates the cache with the new request.
    """
    def serve_request(self, request: int) -> None:
        if self.is_present(request):
            self._frequency_map[request] += 1

        super().serve_request(request)

    """
    Adds an item to a cache, assumes that the cache is not full.
    """
    def add_item(self, item: int) -> None:
        super().add_item(item)
        self._frequency_map[item] = 0

    """
    Removes the item from the cache that has been least frequently used.
    """
    def remove_item_from_cache(self, item: int) -> int:
        self._frequency_map.pop(item)
        return super().remove_item_from_cache(item)

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
