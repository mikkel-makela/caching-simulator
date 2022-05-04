from typing import List, Iterable

from data.loader import load_movie_lens
from factories.cache_factory import get_ftpl_policy
from nodes.cache_system import CacheSystem
from nodes.cache_node import CacheNode
from policies.expert_policies.ftl_policy import ExpertFTLPolicy
from policies.expert_policies.ftpl_policy import ExpertFTPLPolicy
from policies.ftrl_policy import FTRLPolicy
from policies.lfu_policy import LFUPolicy
from policies.lru_policy import LRUPolicy
from policies.policy import Policy
from simulation.simulation_parameters import SimulationParameters
from simulation.simulation_runner import SimulationRunner
from simulation.simulation_statistics import SimulationStatistics, HierarchicalSimulationStatistics
from utilities import print_multi_level_statistics, print_single_level_statistics


class DataPath:
    _shared_path = "./data/raw_data/"
    MOVIE_LENS = f'{_shared_path}ml_25m.csv'


dataset = load_movie_lens(DataPath.MOVIE_LENS)
catalog_size = dataset.catalog_size


def run_single_cache_simulation():
    cache_size = 50
    policies: List[Policy] = [
        LRUPolicy(cache_size),
        LFUPolicy(cache_size),
        FTRLPolicy(cache_size, dataset.catalog_size, len(dataset.trace)),
        # ExpertFTLPolicy(cache_size, [LRUPolicy(cache_size), LFUPolicy(cache_size)]),
        # ExpertFTPLPolicy(cache_size, [LRUPolicy(cache_size), LFUPolicy(cache_size)])
    ]

    runner = SimulationRunner(threads=len(policies))
    parameters = SimulationParameters(catalog_size, dataset.trace, policies)
    statistics: Iterable[SimulationStatistics] = runner.run_simulations(parameters)
    print_single_level_statistics(statistics)


def run_multi_cache_simulation():
    system = CacheSystem([
        CacheNode(5.0, get_ftpl_policy(500), children=[
            CacheNode(3.0, get_ftpl_policy(50)),
            CacheNode(7.0, get_ftpl_policy(50)),
            CacheNode(2.0, get_ftpl_policy(50))
        ]),
        CacheNode(10.0, get_ftpl_policy(500), children=[
            CacheNode(2.0, get_ftpl_policy(50)),
            CacheNode(1.0, get_ftpl_policy(50)),
            CacheNode(3.0, get_ftpl_policy(50))
        ])
    ])
    runner = SimulationRunner(threads=6)
    statistics: HierarchicalSimulationStatistics = runner.run_hierarchical_simulations(system, dataset.trace)
    print_multi_level_statistics(statistics)


def main():
    run_single_cache_simulation()
    # run_multi_cache_simulation()


if __name__ == "__main__":
    main()
