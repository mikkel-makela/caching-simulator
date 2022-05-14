from typing import List

import numpy as np

from system.nodes.node import Node
from system.request import Request


class Client:

    _visible_caches: List[tuple[float, Node]]

    def __init__(self, visible_caches: List[tuple[float, Node]]):
        self._visible_caches = sorted(visible_caches, key=lambda c: c[0])

    def execute_trace(self, trace: np.ndarray) -> None:
        for file in trace:
            self.execute_request(file)

    def execute_request(self, file: int) -> None:
        for reach_cost, cache in self._visible_caches:
            if cache.process_request_and_get_result(Request(file, initial_cost=reach_cost)):
                break
