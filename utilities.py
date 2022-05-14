from typing import List

from matplotlib import pyplot as plt

from simulation.simulation_statistics import HitRatioTree, SimulationStatistics, HierarchicalSimulationStatistics


def get_hit_ratio(hits, misses) -> float:
    assert hits >= 0
    assert misses >= 0
    total = hits + misses
    return 0 if total == 0 else hits / total


def display_single_level_statistics(statistics: List[SimulationStatistics]) -> None:
    for statistic in statistics:
        print(f'=========={statistic.policy}==========')
        print(f'Cumulative hit rate: {round(statistic.hit_ratio * 100, 2)}%')
        plt.plot(statistic.regret, label=statistic.policy)

    plt.ylabel(r"$\frac{R_{T}}{T}$", rotation=0, labelpad=15, fontsize=20)
    plt.xlabel(r"$T$", fontsize=15)
    plt.legend(loc="upper right")
    plt.show()

    plt.figure()
    for statistic in statistics:
        plt.plot(statistic.hit_ratio_list, label=statistic.policy.split(" ")[0])

    plt.ylabel(r"Hit ratio")
    plt.xlabel(r"$T$", fontsize=15)
    plt.legend(loc="upper right")
    plt.show()


def display_multi_level_statistics(statistics: List[HierarchicalSimulationStatistics]) -> None:
    for statistic in statistics:
        display_multi_level_statistic(statistic)
        if statistic.costs is not None:
            plt.plot(statistic.costs, label=statistic.policy)

    plt.ylabel(r"Cost")
    plt.xlabel(r"$T$", fontsize=15)
    plt.legend(loc="upper left")
    plt.show()


def display_multi_level_statistic(statistic: HierarchicalSimulationStatistics) -> None:
    print(f'=================== {statistic.policy} ===================')
    print(f'System-wide cost: {statistic.total_cost}')
    print("Hit Ratio Per Cache")
    print("===================")
    current_level = 0
    queue: List[tuple[HitRatioTree, int]] = [(statistic.hit_ratio_tree, current_level)]
    while len(queue) > 0:
        current, level = queue.pop()
        queue += list(map(lambda c: (c, current_level + 1), current.children))
        if level != current:
            end = "\n"
            current_level = level
        else:
            end = ""
        print(current, end=end)
