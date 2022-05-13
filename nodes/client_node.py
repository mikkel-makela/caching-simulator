from typing import List

import numpy as np

from nodes.node import Node
from nodes.request import Request


class ClientNode:

    _visible_caches: List[tuple[float, Node]]

    def __init__(self, visible_caches: List[tuple[float, Node]]):
        self._visible_caches = sorted(visible_caches, key=lambda c: c[0])

    def execute_trace(self, trace: np.ndarray) -> None:
        for request in trace:
            for reach_cost, cache in self._visible_caches:
                if cache.process_request_and_get_result(Request(request, initial_cost=reach_cost)):
                    break
