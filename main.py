import random
from typing import List

from data.loaders import load_movielens, load_bipartite_movielens
from factories.cache_factory import get_bipartite_systems_from_datasets
from policies.ftrl_policy import FTRLPolicy
from policies.lfu_policy import LFUPolicy
from policies.lru_policy import LRUPolicy
from policies.policy import Policy
from simulation.simulation_parameters import SimulationParameters
from simulation.simulation_runner import SimulationRunner
from simulation.simulation_statistics import SimulationStatistics
from utilities import display_multi_level_statistics, display_single_level_statistics


random.seed(42)


class DataPath:
    _shared_path = "./data/raw_data/"
    MOVIE_LENS = f'{_shared_path}ml_25m.csv'


def run_single_cache_simulation():
    dataset = load_movielens(DataPath.MOVIE_LENS)
    cache_size = 50
    policies: List[Policy] = [
        LRUPolicy(cache_size),
        LFUPolicy(cache_size),
        FTRLPolicy(cache_size, dataset.catalog_size, dataset.trace.size),
        # ExpertFTLPolicy(cache_size, [LRUPolicy(cache_size), LFUPolicy(cache_size)]),
        # ExpertFTPLPolicy(cache_size, [LRUPolicy(cache_size), LFUPolicy(cache_size)])
    ]

    runner = SimulationRunner(threads=len(policies))
    parameters = SimulationParameters(dataset.trace, policies)
    statistics: List[SimulationStatistics] = runner.run_simulations(parameters)
    display_single_level_statistics(statistics)


def run_multi_cache_simulation():
    datasets = load_bipartite_movielens(DataPath.MOVIE_LENS)
    cache_size = 50
    d_regular_degree = 2
    cache_count = 6
    systems = get_bipartite_systems_from_datasets(datasets, cache_size, d_regular_degree, cache_count)
    runner = SimulationRunner(threads=min([len(systems), 10]))
    statistics = runner.run_bipartite_simulations(systems, datasets)
    display_multi_level_statistics(statistics)


def main():
    # run_single_cache_simulation()
    run_multi_cache_simulation()


if __name__ == "__main__":
    main()
