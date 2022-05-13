from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np


class HitMissLogs:
    hits: int = 0
    misses: int = 0


@dataclass
class HitRatioTree:
    children: List[HitRatioTree]
    hit_ratio: float
    policy: str

    def __str__(self):
        return f'{self.policy}: {round(self.hit_ratio, 2)}'


@dataclass
class SimulationStatistics:
    hit_ratio: float
    policy: str
    regret: np.ndarray
    hit_ratio_list: np.ndarray


@dataclass
class HierarchicalSimulationStatistics:
    hit_ratio_tree: HitRatioTree
    total_cost: float

