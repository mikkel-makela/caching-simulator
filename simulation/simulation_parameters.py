import numpy as np

from policies.policy import Policy


class SimulationParameters:
    catalog_size: int
    trace: np.ndarray
    time: int
    policies: [Policy]

    def __init__(self, catalog_size: int, trace: np.ndarray, policies: [Policy], time=None):
        self.catalog_size = catalog_size
        self.trace = trace
        self.time = self.trace.size if time is None else time
        self.policies = policies
