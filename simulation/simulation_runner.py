from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Dict

import numpy as np

from data.loaders import BiPartiteDataset
from policies.network_policies.network_policy import NetworkPolicy
from policies.policy import Policy
from simulation.simulation_parameters import SimulationParameters
from simulation.simulation_statistics import SimulationStatistics, BiPartiteSimulationStatistics
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


def _get_optimal_network_statistics(
        traces: np.ndarray,
        client_cache_connections: List[List[int]],
        cache_size: int,
        caches: int
) -> BiPartiteSimulationStatistics:
    time_horizon = traces[0].size
    rewards = np.zeros(time_horizon)
    cache_trace_map: Dict[int, List[int]] = dict()

    for t in range(time_horizon):
        for client in range(len(client_cache_connections)):
            for cache in client_cache_connections[client]:
                if cache not in cache_trace_map:
                    cache_trace_map[cache] = []
                cache_trace_map[cache].append(traces[client][t])

        reward = 0
        for cache in range(caches):
            trace = np.array(cache_trace_map[cache])
            reward += _get_optimal_static_configuration_hits(trace, cache_size, trace.size)

        rewards[t] = reward / (t + 1)

    return BiPartiteSimulationStatistics(
        policy="OPT",
        rewards=rewards
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


def _execute_system_synchronously(policy: NetworkPolicy, data: BiPartiteDataset) -> np.ndarray:
    clients, time_horizon = data.traces.shape
    rewards = np.zeros(time_horizon)
    for t in range(time_horizon):
        requests = np.zeros(clients)
        for client in range(clients):
            requests[client] = data.traces[client][t]
        policy.update(requests)
        rewards[t] = policy.reward / (t + 1)

    return rewards


def _run_bipartite_simulation(
        policy: NetworkPolicy,
        data: BiPartiteDataset
) -> BiPartiteSimulationStatistics:
    rewards = _execute_system_synchronously(policy, data)
    return BiPartiteSimulationStatistics(
        policy=policy.get_name(),
        rewards=rewards
    )


class SimulationRunner:
    """
    Simulation runner class that is able to run multiple policies concurrently.
    """
    _executor: ThreadPoolExecutor

    def __init__(self, threads: int = 1):
        assert threads >= 1
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
            policies: List[NetworkPolicy],
            data: BiPartiteDataset
    ) -> List[BiPartiteSimulationStatistics]:
        assert len(policies) > 0
        assert data.traces.size > 0

        futures: List[Future] = [
            self._executor.submit(_run_bipartite_simulation, policy, data)
            for policy in policies
        ]

        return list(map(lambda f: f.result(), futures)) + [
            _get_optimal_network_statistics(
                data.traces,
                policies[0].client_cache_connections,
                policies[0].cache_size,
                len(policies)
            )
        ]

