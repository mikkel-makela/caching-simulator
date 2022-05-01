from policies.ftpl_policy import FTPLPolicy
from policies.lfu_policy import LFUPolicy
from policies.lru_policy import LRUPolicy


def get_ftpl_policy(size: int):
    return FTPLPolicy(size, [LRUPolicy(size), LFUPolicy(size)])
