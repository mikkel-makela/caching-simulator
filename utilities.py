from typing import List, Iterable

from simulation.simulation_statistics import HitRatioTree, SimulationStatistics, HierarchicalSimulationStatistics


def print_single_level_statistics(statistics: Iterable[SimulationStatistics]) -> None:
    for statistic in statistics:
        print(f'=========={statistic.policy}==========')
        print(f'Cumulative hit rate: {round(statistic.hit_ratio * 100, 2)}%')


def print_multi_level_statistics(statistics: HierarchicalSimulationStatistics) -> None:
    print(f'System-wide cost: {statistics.total_cost}\n')
    print("Hit Ratio Per Cache")
    print("===================")
    current_level = 0
    queue: List[tuple[HitRatioTree, int]] = [(statistics.hit_ratio_tree, current_level)]
    while len(queue) > 0:
        current, level = queue.pop()
        queue += list(map(lambda c: (c, current_level + 1), current.children))
        if level != current:
            end = "\n"
            current_level = level
        else:
            end = ""
        print(current, end=end)
