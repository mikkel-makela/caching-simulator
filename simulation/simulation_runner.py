from concurrent.futures import ThreadPoolExecutor, Future
from typing import List

import numpy as np

from nodes.cache_system import CacheSystem
from nodes.node import Node
from policies.policy import Policy
from simulation.simulation_parameters import SimulationParameters
from simulation.simulation_statistics import SimulationStatistics, HierarchicalSimulationStatistics, HitRatioTree
from utilities import get_hit_ratio


def _get_optimal_static_configuration_hits(trace: np.ndarray, cache_size: int, time: int) -> int:
    _, counts = np.unique(trace[:time], return_counts=True)
    return int(np.sum(-np.sort(-counts)[:cache_size]))


def _run_simulation(
        trace: np.ndarray,
        time_horizon: int,
        policy: Policy
) -> SimulationStatistics:
    assert time_horizon > 0
    assert trace.size > 0
    assert policy is not None

    hits = misses = 0
    regret = np.zeros(time_horizon)
    hit_ratio = np.zeros(time_horizon)
    for time, request in enumerate(trace[0:time_horizon], start=1):
        if policy.is_present(request):
            hits += 1
        else:
            misses += 1
        regret[time - 1] = (_get_optimal_static_configuration_hits(trace, policy.cache.size, time) - hits) / time
        hit_ratio[time - 1] = get_hit_ratio(hits, misses)
        policy.update(request)

    return SimulationStatistics(
        get_hit_ratio(hits, misses),
        policy.get_name(),
        regret,
        hit_ratio
    )


def _get_hierarchical_statistics(cache_system: CacheSystem) -> HierarchicalSimulationStatistics:
    def get_trees(nodes: List[Node], level=1) -> List[HitRatioTree]:
        return list(
            map(
                lambda n: HitRatioTree(
                    get_trees(n.children, level + 1),
                    get_hit_ratio(n.hit_miss_logs.hits, n.hit_miss_logs.misses),
                    f'{n.policy.get_name()}, Level {level}'
                ),
                nodes
            )
        )

    return HierarchicalSimulationStatistics(
        HitRatioTree(get_trees(cache_system.main_server.children), 1.0, "Main Storage, Level 0"),
        cache_system.get_absorbed_cost()
    )


class SimulationRunner:
    _executor: ThreadPoolExecutor

    def __init__(self, threads: int = 1):
        assert threads >= 1
        self._executor = ThreadPoolExecutor(max_workers=threads)

    def run_simulations(self, parameters: SimulationParameters) -> List[SimulationStatistics]:
        assert parameters is not None
        assert parameters.policies is not None and len(parameters.policies) > 0

        futures: List[Future] = [
            self._executor.submit(_run_simulation, parameters.trace, parameters.time, policy)
            for policy in parameters.policies
        ]

        return list(map(lambda f: f.result(), futures))

    def run_bipartite_simulations(
            self,
            cache_system: CacheSystem,
            datasets: np.ndarray
    ) -> HierarchicalSimulationStatistics:
        assert cache_system is not None
        assert len(cache_system.clients) == datasets.size

        futures: List[Future] = [
            self._executor.submit(client.execute_trace, dataset.trace)
            for dataset, client in zip(datasets, cache_system.clients)
        ]

        for future in futures:
            future.result()

        return _get_hierarchical_statistics(cache_system)

