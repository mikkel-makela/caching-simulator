from __future__ import annotations
from dataclasses import dataclass

import numpy as np


class HitMissLogs:
    hits: int = 0
    misses: int = 0


@dataclass
class Statistics:
    policy: str


@dataclass
class SimulationStatistics(Statistics):
    hit_ratio: float
    regret: np.ndarray
    hit_ratios: np.ndarray


@dataclass
class HierarchicalSimulationStatistics(Statistics):
    total_cost: float
    costs: np.ndarray or None
    hit_ratios_for_caches: np.ndarray
    hit_ratios_t: np.ndarray or None

