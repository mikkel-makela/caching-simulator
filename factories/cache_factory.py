from policies.expert_policies.ftpl_policy import ExpertFTPLPolicy
from policies.lfu_policy import LFUPolicy
from policies.lru_policy import LRUPolicy


def get_ftpl_policy(size: int):
    return ExpertFTPLPolicy(size, [LRUPolicy(size), LFUPolicy(size)])
