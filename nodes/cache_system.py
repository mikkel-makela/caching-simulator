from typing import List

from nodes.cache_node import CacheNode


def get_absorbed_cost(node: CacheNode):
    return node.absorbed_system_cost + sum(map(lambda child: get_absorbed_cost(child), node.children))


def get_leaf_nodes(nodes: List[CacheNode]) -> List[CacheNode]:
    leafs = []
    for node in nodes:
        leafs += [node] if len(node.children) == 0 else get_leaf_nodes(node.children)
    return leafs


class CacheSystem:
    """
    To use for simulations, keep a list of child nodes and make them serve requests.
    """

    first_layer_nodes: List[CacheNode]

    def __init__(self, first_layer_nodes: List[CacheNode]):
        self.first_layer_nodes = first_layer_nodes

    def get_absorbed_cost(self) -> float:
        return sum(map(lambda node: get_absorbed_cost(node), self.first_layer_nodes))

    def get_leaf_nodes(self) -> List[CacheNode]:
        return get_leaf_nodes(self.first_layer_nodes)

