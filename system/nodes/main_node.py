import threading
from typing import List

from system.nodes.cache_node import CacheNode
from system.nodes.node import Node
from system.request import Request


class MainNode(Node):

    lock: threading.Lock

    def __init__(self, children: List[CacheNode]):
        super().__init__(children)
        self.lock = threading.Lock()

    def process_request_and_get_result(self, request: Request):
        self.lock.acquire()
        self.absorbed_system_cost += request.current_cost
        self.lock.release()

    def get_hit_ratio(self) -> float:
        return 1.0
