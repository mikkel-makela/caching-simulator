from dataclasses import dataclass
from typing import List

from system.client import Client
from system.nodes.node import Node
from utilities import get_hit_ratio


@dataclass
class CacheSystem:
    """
    Datastructure that combines the caches and clients of a system.
    """

    clients: List[Client]
    policy: str
    caches: List[Node]

    def get_absorbed_cost(self) -> float:
        return sum(map(lambda client: client.absorbed_cost, self.clients))

    def get_cumulative_hit_ratio(self) -> float:
        hits = sum(map(lambda node: node.hit_miss_logs.hits, self.caches))
        misses = sum(map(lambda node: node.hit_miss_logs.misses, self.caches))
        return get_hit_ratio(hits, misses)
