from dataclasses import dataclass
from typing import List

from nodes.client_node import ClientNode
from nodes.main_node import MainNode
from nodes.node import Node


def get_absorbed_cost(node: Node):
    return node.absorbed_system_cost + sum(map(lambda child: get_absorbed_cost(child), node.children))


@dataclass
class CacheSystem:
    """
    Datastructure that stores the main server of a system and all its clients.
    """

    main_server: MainNode
    clients: List[ClientNode]

    def get_absorbed_cost(self) -> float:
        return get_absorbed_cost(self.main_server)
