from __future__ import annotations

import threading
from typing import List

from nodes.request import Request
from policies.eviction_policy import EvictionPolicy
from simulation.simulation_statistics import HitMissLogs


class CacheNode:

    _parent_reach_cost: float
    policy: EvictionPolicy
    parent: CacheNode or None = None  # None represents the main data store
    children: List[CacheNode]
    absorbed_system_cost: float
    hit_miss_logs: HitMissLogs
    lock: threading.Lock  # Used to make request handling thread safe

    def __init__(self, parent_reach_cost: float, policy: EvictionPolicy, children=None):
        self._parent_reach_cost = parent_reach_cost
        self.policy = policy
        self.children = [] if children is None else children
        self.lock = threading.Lock()
        self.hit_miss_logs = HitMissLogs()
        self.absorbed_system_cost = 0
        for child in self.children:
            child.parent = self

    def process_request(self, request: Request):
        self.lock.acquire()
        if not self.policy.is_present(request.catalog_item) and self.parent is None:
            # Parent is the main cache, current cost and the final fetch cost need to be observed
            self.absorbed_system_cost += request.current_cost + self._parent_reach_cost
        elif self.policy.is_present(request.catalog_item):
            # Request can be served, so all cost gets absorbed
            self.absorbed_system_cost += request.current_cost
        else:
            self.parent.process_request(request.get_with_incremented_cost(self._parent_reach_cost))

        self.update_logs(request.catalog_item)
        self.policy.serve_request(request.catalog_item)
        self.lock.release()

    def update_logs(self, item: int):
        if self.policy.is_present(item):
            self.hit_miss_logs.hits += 1
        else:
            self.hit_miss_logs.misses += 1
