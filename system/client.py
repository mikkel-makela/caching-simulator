from dataclasses import dataclass
from typing import List

import numpy as np

from system.nodes.node import Node
from system.requests import Request


@dataclass
class Client:

    cache: Node
    cache_reach_cost: float
    connections: List[Node]
    absorbed_cost: float = 0

    def execute_trace(self, trace: np.ndarray) -> None:
        for file in trace:
            self.execute_request(file)

    def execute_request(self, file: int) -> None:
        response = self.cache.process_request_and_get_result(Request(file, initial_cost=self.cache_reach_cost))
        assert response.was_hit
        self.absorbed_cost += response.cost
