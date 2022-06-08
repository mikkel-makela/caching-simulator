from __future__ import annotations

from typing import List

from system.requests import Request, Response


class Node:

    absorbed_system_cost: float
    visible_nodes: List[tuple[float, Node]]

    def __init__(self, visible_nodes: List[tuple[float, Node]] = None):
        self.absorbed_system_cost = 0
        self.visible_nodes = [] if visible_nodes is None else visible_nodes
        self.sort_visible_nodes()

    def process_request_and_get_result(self, request: Request) -> Response:
        pass

    def get_hit_ratio(self) -> float:
        pass

    def add_edge(self, weight: float, to_node: Node) -> None:
        assert to_node is not None
        assert weight >= 0
        self.visible_nodes.append((weight, to_node))
        self.sort_visible_nodes()

    def sort_visible_nodes(self):
        self.visible_nodes = list(sorted(self.visible_nodes, key=lambda pair: pair[0]))
