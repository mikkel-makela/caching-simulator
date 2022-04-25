import numpy as np


class SimulationStatistics:

    hit_ratio_array: np.ndarray
    hit_ratio: float
    policy: str

    def __init__(self, hit_ratio_array: np.ndarray, hit_ratio: float, policy: str):
        self.hit_ratio_array = hit_ratio_array
        self.hit_ratio = hit_ratio
        self.policy = policy

