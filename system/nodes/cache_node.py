from __future__ import annotations

import threading
from typing import List

from system.nodes.node import Node
from system.request import Request
from policies.policy import Policy
from simulation.simulation_statistics import HitMissLogs
from utilities import get_hit_ratio


class CacheNode(Node):

    _parent_reach_cost: float
    policy: Policy
    parent: Node
    hit_miss_logs: HitMissLogs
    lock: threading.Lock

    def __init__(self, parent_reach_cost: float, policy: Policy, children: List[CacheNode] or None = None):
        super().__init__(children)
        self._parent_reach_cost = parent_reach_cost
        self.policy = policy
        self.lock = threading.Lock()
        self.hit_miss_logs = HitMissLogs()

    """
    Tries to process the request. Returns whether it was successful.
    """
    def process_request_and_get_result(self, request: Request) -> bool:
        self.lock.acquire()
        if self.policy.is_present(request.catalog_item):
            return self.handle_file_present(request)
        else:
            return self.handle_file_missing(request)

    def handle_file_present(self, request: Request) -> bool:
        self.hit_miss_logs.hits += 1
        self.absorbed_system_cost += request.current_cost
        self.update_policy_and_release_lock(request.catalog_item)
        return True

    def handle_file_missing(self, request: Request) -> bool:
        self.hit_miss_logs.misses += 1
        self.update_policy_and_release_lock(request.catalog_item)
        if self.parent is None:
            return False
        return self.parent.process_request_and_get_result(request.get_with_incremented_cost(self._parent_reach_cost))

    def update_policy_and_release_lock(self, file: int):
        self.policy.update(file)
        self.lock.release()

    def get_hit_ratio(self) -> float:
        return get_hit_ratio(self.hit_miss_logs.hits, self.hit_miss_logs.misses)
