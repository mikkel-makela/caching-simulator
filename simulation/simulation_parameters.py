import numpy as np

from policies.policy import Policy


class SimulationParameters:
    trace: np.ndarray
    time: int
    policies: [Policy]

    def __init__(self, trace: np.ndarray, policies: [Policy], time=None):
        self.trace = trace
        self.time = self.trace.size if time is None else time
        self.policies = policies
