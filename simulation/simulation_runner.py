from concurrent.futures import ThreadPoolExecutor, Future
from typing import Iterable, List

import numpy as np

from nodes.cache_system import CacheSystem
from nodes.cache_node import CacheNode
from nodes.request import Request
from policies.policy import Policy
from simulation.simulation_parameters import SimulationParameters
from simulation.simulation_statistics import SimulationStatistics, HierarchicalSimulationStatistics, HitRatioTree


def _get_hit_ratio(hits, misses) -> float:
    assert hits >= 0
    assert misses >= 0
    total = hits + misses
    return 0 if total == 0 else hits / total


def _run_simulation(
        catalog_size: int,
        trace: np.ndarray,
        time_horizon: int,
        policy: Policy
) -> SimulationStatistics:
    assert time_horizon > 0
    assert trace.size > 0
    assert policy is not None

    hit_miss_logs = np.zeros((catalog_size + 1, 2))
    for request in trace[0:time_horizon]:
        hit_miss_logs[request][0 if policy.is_present(request) else 1] += 1
        policy.update(request)

    return SimulationStatistics(
        np.array(list(map(lambda x: _get_hit_ratio(x[0], x[1]), hit_miss_logs))),
        sum(log[0] for log in hit_miss_logs) / time_horizon,
        policy.get_name()
    )


def _run_hierarchical_simulation(trace: np.ndarray, cache: CacheNode) -> None:
    for request in trace:
        cache.process_request(Request(request))


def _get_hierarchical_statistics(cache_system: CacheSystem) -> HierarchicalSimulationStatistics:
    def get_trees(nodes: List[CacheNode], level=1) -> List[HitRatioTree]:
        return list(
            map(
                lambda n: HitRatioTree(
                    get_trees(n.children, level + 1),
                    _get_hit_ratio(n.hit_miss_logs.hits, n.hit_miss_logs.misses),
                    f'{n.policy.get_name()}, Level {level}'
                ),
                nodes
            )
        )

    return HierarchicalSimulationStatistics(
        HitRatioTree(get_trees(cache_system.first_layer_nodes), 1.0, "Main Storage, Level 0"),
        cache_system.get_absorbed_cost()
    )


class SimulationRunner:
    _executor: ThreadPoolExecutor

    def __init__(self, threads: int = 1):
        assert threads >= 1
        self._executor = ThreadPoolExecutor(max_workers=threads)

    def run_simulations(self, parameters: SimulationParameters) -> Iterable[SimulationStatistics]:
        assert parameters is not None
        assert parameters.policies is not None and len(parameters.policies) > 0

        futures: List[Future] = [
            self._executor.submit(_run_simulation, parameters.catalog_size, parameters.trace, parameters.time, policy)
            for policy in parameters.policies
        ]

        return map(lambda f: f.result(), futures)

    def run_hierarchical_simulations(
            self,
            cache_system: CacheSystem,
            trace: np.ndarray
    ) -> HierarchicalSimulationStatistics:
        assert cache_system is not None
        assert trace.size > 0
        caches = cache_system.get_leaf_nodes()
        traces = np.array_split(trace, len(caches))

        futures: List[Future] = [
            self._executor.submit(_run_hierarchical_simulation, trace, cache)
            for trace, cache in zip(traces, caches)
        ]

        for future in futures:
            future.result()

        return _get_hierarchical_statistics(cache_system)

