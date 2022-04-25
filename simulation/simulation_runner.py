from concurrent.futures import ThreadPoolExecutor, Future
from typing import Iterable

import numpy as np

from policies.policy import Policy
from simulation.simulation_parameters import SimulationParameters
from simulation.simulation_statistics import SimulationStatistics


def get_hit_ratio(hits, misses) -> float:
    assert hits >= 0
    assert misses >= 0
    total = hits + misses
    return 0 if total == 0 else hits / total


def _run_simulation(catalog_size: int,
                    trace: np.ndarray,
                    time_horizon: int,
                    policy: Policy
                    ) -> SimulationStatistics:
    assert time_horizon > 0
    assert trace.size > 0
    assert policy is not None

    hit_miss_logs = np.zeros((catalog_size + 1, 2))
    for request in trace[0:time_horizon]:
        if policy.is_present(request):
            hit_miss_logs[request][0] += 1
        else:
            hit_miss_logs[request][1] += 1
        policy.serve_request(request)

    return SimulationStatistics(
        np.array(list(map(lambda x: get_hit_ratio(x[0], x[1]), hit_miss_logs))),
        sum(log[0] for log in hit_miss_logs) / time_horizon,
        policy.get_name()
    )


class SimulationRunner:
    _executor: ThreadPoolExecutor

    def __init__(self, threads: int = 1):
        assert threads >= 1
        self._executor = ThreadPoolExecutor(max_workers=threads)

    def run_simulations(self, parameters: SimulationParameters) -> Iterable[SimulationStatistics]:
        assert parameters is not None
        assert parameters.policies is not None and len(parameters.policies) > 0

        futures: [Future] = [
            self._executor.submit(_run_simulation, parameters.catalog_size, parameters.trace, parameters.time, policy)
            for policy in parameters.policies
        ]

        return map(lambda f: f.result(), futures)
