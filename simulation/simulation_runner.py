from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import List

import numpy as np

from system.cache_system import CacheSystem
from system.client import Client
from policies.policy import Policy
from simulation.simulation_parameters import SimulationParameters
from simulation.simulation_statistics import SimulationStatistics, HierarchicalSimulationStatistics
from utilities import get_hit_ratio


def _get_optimal_static_configuration_hits(trace: np.ndarray, cache_size: int, time: int) -> int:
    _, counts = np.unique(trace[:time], return_counts=True)
    return int(np.sum(-np.sort(-counts)[:cache_size]))


def _get_optimal_static_statistics(trace: np.ndarray, time_horizon: int, cache_size: int) -> SimulationStatistics:
    hit_ratio = np.zeros(time_horizon)
    for time, request in enumerate(trace[0:time_horizon], start=1):
        hit_ratio[time - 1] = _get_optimal_static_configuration_hits(trace, cache_size, time) / time
    return SimulationStatistics(
        "OPT",
        hit_ratio[time_horizon - 1],
        np.zeros(time_horizon),
        hit_ratio
    )


def _run_single_cache_simulation(
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
        policy.get_name(),
        get_hit_ratio(hits, misses),
        regret,
        hit_ratio
    )


def _get_hierarchical_statistics(
        cache_system: CacheSystem,
        costs: np.ndarray or None = None,
        hit_ratios: np.ndarray or None = None
) -> HierarchicalSimulationStatistics:
    hit_ratios_for_caches = np.array(
        list(
            map(
                lambda cache: get_hit_ratio(cache.hit_miss_logs.hits, cache.hit_miss_logs.misses),
                cache_system.caches
            )
        )
    )
    return HierarchicalSimulationStatistics(
        cache_system.policy,
        cache_system.get_absorbed_cost(),
        costs,
        hit_ratios_for_caches,
        hit_ratios
    )


def _execute_system_synchronously(
        system: CacheSystem,
        datasets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    @dataclass
    class ClientTrace:
        client: Client
        trace: np.ndarray
        execution_step: int

        def execute_next(self):
            self.client.execute_request(self.trace[self.execution_step])
            self.execution_step += 1

        def is_ready(self) -> bool:
            return self.execution_step == self.trace.size

    client_traces: List[ClientTrace] = list(
        map(lambda pair: ClientTrace(pair[0], pair[1].trace, 0), zip(system.clients, datasets))
    )

    costs = np.zeros(sum(map(lambda ds: ds.trace.size, datasets)))
    hit_ratio = np.zeros(costs.size)
    t = 0
    while len(client_traces) > 0:
        for client_trace in client_traces:
            if client_trace.is_ready():
                client_traces.remove(client_trace)
                break
            client_trace.execute_next()
            costs[t] = system.get_absorbed_cost()
            hit_ratio[t] = system.get_cumulative_hit_ratio()
            t += 1

    return costs, hit_ratio


class SimulationRunner:
    """
    Simulation runner class that is able to run multiple policies concurrently.
    """
    _executor: ThreadPoolExecutor
    _use_concurrency: bool

    def __init__(self, use_concurrency: bool = False, threads: int = 1):
        """
        Initializes the runner with specified thread count. If concurrency is enabled, runs all clients in
        concurrently in a bi-partite setting, but is unable to record system costs at each iteration.

        :param use_concurrency: Whether bi-partite simulations should be run concurrently.
        :param threads: How many threads to allocate.
        """
        assert threads >= 1
        self._use_concurrency = use_concurrency
        self._executor = ThreadPoolExecutor(max_workers=threads)

    def run_simulations(self, parameters: SimulationParameters) -> List[SimulationStatistics]:
        assert parameters is not None
        assert parameters.policies is not None and len(parameters.policies) > 0

        futures: List[Future] = [
            self._executor.submit(_run_single_cache_simulation, parameters.trace, parameters.time, policy)
            for policy in parameters.policies
        ]

        return list(map(lambda f: f.result(), futures)) + [
            _get_optimal_static_statistics(parameters.trace, parameters.time, parameters.policies[0].cache.size)
        ]

    def run_bipartite_simulations(
            self,
            cache_systems: List[CacheSystem],
            datasets: np.ndarray
    ) -> List[HierarchicalSimulationStatistics]:
        assert len(cache_systems) > 0
        assert datasets.size > 0

        futures: List[Future] = [
            self._executor.submit(self._run_bipartite_simulation, system, datasets)
            for system in cache_systems
        ]

        return list(map(lambda f: f.result(), futures))

    def _run_bipartite_simulation(
            self,
            cache_system: CacheSystem,
            datasets: np.ndarray
    ) -> HierarchicalSimulationStatistics:
        costs = hit_ratios = None
        if self._use_concurrency:
            self._execute_system_concurrently(cache_system.clients, datasets)
        else:
            costs, hit_ratios = _execute_system_synchronously(cache_system, datasets)

        return _get_hierarchical_statistics(cache_system, costs, hit_ratios)

    def _execute_system_concurrently(
            self,
            clients: List[Client],
            datasets: np.ndarray
    ) -> None:
        futures: List[Future] = [
            self._executor.submit(client.execute_trace, dataset.trace)
            for dataset, client in zip(datasets, clients)
        ]

        for future in futures:
            future.result()
