from __future__ import annotations

from typing import List

from nodes.request import Request


class Node:

    absorbed_system_cost: float
    children: List[Node]

    def __init__(self, children: List[Node]):
        self.absorbed_system_cost = 0
        self.children = children
        for child in children:
            child.parent = self

    def process_request_and_get_result(self, request: Request) -> bool:
        pass

    def get_hit_ratio(self) -> float:
        pass
