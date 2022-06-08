from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import minimize


@dataclass
class LeadCacheSolverParams:
    theta: np.ndarray
    client_cache_map: List[List[int]]
    cache_size: int
    catalog_size: int
    cache_count: int


def get_opt_lead_cache(params: LeadCacheSolverParams) -> np.ndarray:
    """
    Gets the optimal non-integral cache configuration.

    :param params: Cache solver parameters
    :return: Optimal non-integral configuration
    """
    cache_shape = (params.cache_count, params.catalog_size)

    def objective(cache_configuration: np.ndarray) -> float:
        cache_configuration = cache_configuration.reshape(cache_shape)

        def get_z_i(connected_caches: np.ndarray) -> float:
            """
            Computes the virtual action for user i and file f.
            Needs to happen per user, because cache configurations vary

            :param connected_caches: caches connected to user i
            :return: Virtual action for client and file
            """
            return np.minimum(np.ones(cache_configuration[0].size), cache_configuration[connected_caches].sum())

        """
        The objective function that we minimize.

        :param cache_configuration: current cache configuration to use in the computation
        :return: the function value
        """
        return -sum(
            map(
                lambda client: np.sum(
                    np.maximum(
                        np.zeros(params.theta[client].size),
                        params.theta[client]
                    ) * get_z_i(np.array(params.client_cache_map[client]))
                ),
                range(len(params.client_cache_map))
            )
        )

    def cache_size_constraint(cache: np.ndarray) -> float:
        """
        Returns 0 if every cache configuration sums up to cache size.

        :param cache: The current cache
        :return: Sum of all configurations - cache sizes
        """
        return params.cache_size - np.sum(cache)

    def is_integral(configuration: np.ndarray, cache: int, file: int) -> bool:
        return abs(round(configuration[cache][file]) - configuration[cache][file]) < 0.0001

    def round_and_check_is_integral(configuration: np.ndarray) -> bool:
        for c in range(configuration.shape[0]):
            for f in range(configuration.shape[1]):
                if not is_integral(configuration, c, f):
                    return False
                else:
                    configuration[c][f] = round(configuration[c][f])
        return True

    def get_updated_configuration(configuration: np.ndarray, cache: int, non_integrals: List[int]) -> np.ndarray:

        def round_objective(y: np.ndarray) -> float:
            total = 0
            for i, o in enumerate(params.theta):
                o_i = np.maximum(np.zeros(o.size), o)
                for f, o_i_f in enumerate(o_i):
                    connected_product = 1
                    for connected_cache in params.client_cache_map[i]:
                        connected_product *= 1 - y[connected_cache][f]
                    total += o_i_f * (1 - connected_product)
            return total

        assert len(non_integrals) == 2
        non_integral_cache = configuration[cache]
        y_1, y_2 = non_integrals[0], non_integrals[1]
        y_1_value, y_2_value = non_integral_cache[y_1], non_integral_cache[y_2]
        e_1 = min([y_1_value, 1 - y_2_value])
        e_2 = min([1 - y_1_value, y_2_value])
        a = np.copy(configuration)
        b = np.copy(configuration)
        a[cache][y_1] = y_1_value - e_1
        a[cache][y_2] = y_2_value + e_1
        b[cache][y_1] = y_1_value + e_2
        b[cache][y_2] = y_2_value - e_2
        return max([a, b], key=lambda config: round_objective(config))

    def pipage_round(configuration: np.ndarray) -> np.ndarray:
        while not round_and_check_is_integral(configuration):
            # print(configuration)
            for c in range(configuration.shape[0]):
                fractionals = []
                for f in range(configuration.shape[1]):
                    if not is_integral(configuration, c, f):
                        fractionals.append(f)
                    if len(fractionals) == 2:
                        configuration = get_updated_configuration(configuration, c, fractionals)
                        break

        return configuration

    flattened_configuration_size = params.catalog_size * params.cache_count
    initial = np.ones(flattened_configuration_size)
    bounds = tuple([(0, 1)] * flattened_configuration_size)
    constraints = list(map(lambda c: {
            'type': 'ineq',
            'fun': lambda config: cache_size_constraint(
                config[(params.catalog_size * c):(params.catalog_size * (c + 1))]
            )
        }, range(params.cache_count)))
    solution = minimize(objective, initial, bounds=bounds, constraints=constraints, method="SLSQP")
    return pipage_round(solution.x.reshape(cache_shape))
