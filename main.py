from typing import List, Iterable

from data.loader import load_movie_lens
from policies.ftl_policy import FTLPolicy
from policies.ftpl_policy import FTPLPolicy
from policies.lfu_policy import LFUPolicy
from policies.lru_policy import LRUPolicy
from policies.policy import Policy
from simulation.simulation_parameters import SimulationParameters
from simulation.simulation_runner import SimulationRunner
from simulation.simulation_statistics import SimulationStatistics

cache_size = 50
policies: List[Policy] = [
    LRUPolicy(cache_size),
    LFUPolicy(cache_size),
    FTLPolicy(cache_size, [LRUPolicy(cache_size), LFUPolicy(cache_size)]),
    FTPLPolicy(
        cache_size,
        [
            LRUPolicy(cache_size),
            LFUPolicy(cache_size)
        ]
    )
]
dataset = load_movie_lens("./data/raw_data/ml_25m.csv")


def main():
    runner = SimulationRunner(threads=len(policies))
    parameters = SimulationParameters(dataset.catalog_size, dataset.trace, policies)
    print(f'Running simulation with cache that is {cache_size / dataset.catalog_size}% of catalog...')
    statistics: Iterable[SimulationStatistics] = runner.run_simulations(parameters)
    for statistic in statistics:
        print(f'========{statistic.policy}==========')
        print(f'Cumulative hit rate: {round(statistic.hit_ratio * 100, 2)}%')
        # print(f'Hit ratio for each request: {statistic.hit_ratio_array}')


if __name__ == "__main__":
    main()
