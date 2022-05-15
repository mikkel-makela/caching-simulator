import numpy as np

from policies.policy import Policy


class FTRLPolicy(Policy):
    """
    (Non-expert) Follow the Regularized Leader policy.

    TODO: file_request_counts array becomes infeasibly large for big catalogs
    Solution: LFU style approach where we store a subset of requests
    """

    _file_request_counts: np.ndarray
    _capacity: int
    _permutation_constant: float

    def __init__(self, capacity: int, catalog_size: int, time_horizon: int):
        super().__init__(capacity)
        self._capacity = capacity
        self._file_request_counts = np.zeros(catalog_size + 1)
        self._permutation_constant = \
            (1 / (4 * np.pi * np.log(catalog_size)) ** (1.0 / 4.0)) * np.sqrt(time_horizon / capacity)
        self.cache = self.get_updated_cache()

    @staticmethod
    def get_name() -> str:
        return "FTRL"

    """
    Updates the request counts and loads a new cache based on it.
    """
    def update(self, request: int) -> None:
        super().update(request)
        self._file_request_counts[request] += 1
        self.cache = self.get_updated_cache()

    """
    Gets a new cache configuration by selecting the most requested items, where the request count
    is the actual request count added with a random value from a normal distribution multiplied by
    the permutation constant.
    """
    def get_updated_cache(self) -> np.ndarray:
        perturbed_counts = self._file_request_counts + np.random.normal(
            loc=0,
            scale=self._permutation_constant,
            size=self._file_request_counts.size
        )
        return np.argsort(-perturbed_counts)[:self._capacity]

    """
    Resets the cache, deleting all entries.
    """
    def reset(self) -> None:
        super().reset()
        self._file_request_counts = np.zeros(self._file_request_counts.size)
