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
        self.regret = None

    @staticmethod
    def get_name() -> str:
        pass

    """
    Updates the cache based on the new request.
    """
    def update(self, request: int) -> None:
        self.advance_time()

    """
    Resets the cache, deleting all entries.
    """
    def reset(self) -> None:
        self.cache = np.empty(len(self.cache), dtype=object)
        self.time = 1

    """
    Checks if a file is present in the cache.
    """
    def is_present(self, file: int) -> bool:
        return file in self.cache

    """
    Checks if the cache is full, i.e if it has any None elements which represent free slots.
    """
    def is_full(self) -> bool:
        return None not in self.cache

    """
    Advances time.
    """
    def advance_time(self) -> None:
        self.time += 1
