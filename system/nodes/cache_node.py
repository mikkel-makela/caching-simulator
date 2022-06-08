from __future__ import annotations

import threading
from typing import List

from system.nodes.node import Node
from system.requests import Request, Response
from policies.policy import Policy
from simulation.simulation_statistics import HitMissLogs
from utilities import get_hit_ratio


class CacheNode(Node):
    """
    A regular cache node in the caching system. Connected to some nodes that it can proxy requests to.
    """

    policy: Policy
    hit_miss_logs: HitMissLogs
    lock: threading.Lock

    def __init__(self, policy: Policy, visible_nodes: List[tuple[float, Node]] = None):
        super().__init__(visible_nodes)
        self.policy = policy
        self.lock = threading.Lock()
        self.hit_miss_logs = HitMissLogs()

    """
    Tries to process the request. Returns whether it was successful.
    """
    def process_request_and_get_result(self, request: Request) -> Response:
        if self.check_and_update_cache(request.catalog_item):
            self.hit_miss_logs.hits += 1
            return request.build_response(True)

        self.hit_miss_logs.misses += 1
        return self.proxy(request) if self.should_proxy() else request.build_response(False)

    def should_proxy(self) -> bool:
        return len(self.visible_nodes) > 0

    def proxy(self, request: Request) -> Response:
        """
        Routes the request to some visible cache.

        :param request: The request to be routed
        :return: Whether the file was found
        """
        for cost, node in self.visible_nodes:
            response = node.process_request_and_get_result(request.get_with_incremented_cost(cost))
            if response.was_hit:
                return response
            else:
                request.cost = response.cost

        return request.build_response(False)

    def check_and_update_cache(self, file: int) -> bool:
        self.lock.acquire()
        was_present = self.policy.is_present(file)
        self.policy.update(file)
        self.lock.release()
        return was_present

    def get_hit_ratio(self) -> float:
        return get_hit_ratio(self.hit_miss_logs.hits, self.hit_miss_logs.misses)
