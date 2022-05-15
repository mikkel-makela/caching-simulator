from dataclasses import dataclass
from typing import List

from system.client import Client
from system.nodes.main_node import MainNode
from system.nodes.node import Node
from utilities import get_hit_ratio


def get_absorbed_cost(node: Node):
    return node.absorbed_system_cost + sum(map(lambda child: get_absorbed_cost(child), node.children))


def get_all_hits(nodes: List[Node]) -> int:
    return sum(map(lambda node: node.hit_miss_logs.hits + get_all_hits(node.children), nodes))


def get_all_misses(nodes: List[Node]) -> int:
    return sum(map(lambda node: node.hit_miss_logs.misses + get_all_misses(node.children), nodes))


@dataclass
class CacheSystem:
    """
    Datastructure that stores the main server of a system and all its clients.
    """

    main_server: MainNode
    clients: List[Client]
    policy: str

    def get_absorbed_cost(self) -> float:
        return get_absorbed_cost(self.main_server)

    def get_cumulative_hit_ratio(self) -> float:
        caches = self.main_server.children
        return get_hit_ratio(get_all_hits(caches), get_all_misses(caches))
