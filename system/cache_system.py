from dataclasses import dataclass
from typing import List

from system.client import Client
from system.nodes.main_node import MainNode
from system.nodes.node import Node


def get_absorbed_cost(node: Node):
    return node.absorbed_system_cost + sum(map(lambda child: get_absorbed_cost(child), node.children))


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
