import numpy as np

from policies.policy import Policy


class FTPLPolicy(Policy):
    """
    (Non-expert) Follow the Perturbed Leader policy.
    """

    _file_request_counts: np.ndarray
    _capacity: int
    _permutation_constant: float
    _discount_rate: float
    _addition: float

    def __init__(self, capacity: int, catalog_size: int, time_horizon: int, discount_rate: float = 1):
        super().__init__(capacity)
        self._capacity = capacity
        self._discount_rate = discount_rate
        self._file_request_counts = np.zeros(catalog_size + 1)
        self._permutation_constant = \
            (1 / (4 * np.pi * np.log(catalog_size)) ** (1.0 / 4.0)) * np.sqrt(time_horizon / capacity)
        self._addition = 1
        self.cache = self.get_updated_cache()

    def get_name(self) -> str:
        return f'FTPL, d={self._discount_rate}'

    """
    Updates the request counts and loads a new cache based on it.
    """
    def update(self, request: int) -> None:
        super().update(request)
        self.update_request_counts(request)
        self.cache = self.get_updated_cache()

    def update_request_counts(self, request: int):
        if self.time == 1:
            self._file_request_counts[request] += 1.0
            return

        self._file_request_counts *= self._discount_rate
        self._file_request_counts[request] += self.time - self._discount_rate * (self.time - 1)

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
