from typing import List

from matplotlib import pyplot as plt

from simulation.simulation_statistics import SimulationStatistics, BiPartiteSimulationStatistics


def get_hit_ratio(hits, misses) -> float:
    assert hits >= 0
    assert misses >= 0
    total = hits + misses
    return 0 if total == 0 else hits / total


def display_single_level_statistics(statistics: List[SimulationStatistics], print_results: bool = True) -> None:
    for statistic in statistics:
        if print_results:
            print(f'{statistic.policy} hit rate: {round(statistic.hit_ratio * 100, 2)}%')
        plt.plot(statistic.regret, label=statistic.policy)

    plt.ylabel(r"$\frac{R_{T}}{T}$", rotation=0, labelpad=15, fontsize=20)
    plt.xlabel(r"$T$", fontsize=15)
    plt.legend(loc="upper right")
    plt.ylim([0, 0.4])
    plt.show()

    plt.figure()
    best_hit_ratio = 0
    for statistic in statistics:
        plt.plot(statistic.hit_ratios, label=statistic.policy)
        best_hit_ratio = max([best_hit_ratio, statistic.hit_ratio])

    plt.ylabel(r"Hit ratio")
    plt.xlabel(r"$T$", fontsize=15)
    plt.legend(loc="upper left")
    plt.ylim([0, min([best_hit_ratio + 0.2, 1])])
    plt.show()


def display_multi_level_statistics(statistics: List[BiPartiteSimulationStatistics]) -> None:
    for statistic in statistics:
        plt.plot(statistic.rewards, label=statistic.policy)

    plt.ylabel(r"$\frac{q_{T}}{T}$", rotation=0, labelpad=15, fontsize=20)
    plt.xlabel(r"$T$", fontsize=15)
    plt.legend(loc="upper left")
    plt.ylim([0, max(statistics, key=lambda s: s.rewards[-1]).rewards[-1] * 1.5])
    plt.show()

