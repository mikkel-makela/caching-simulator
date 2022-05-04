import numpy as np

from policies.expert_policies.expert_adapter.expert_adapter import ExpertAdaptedPolicy
from policies.expert_policies.ftl_policy import ExpertFTLPolicy
from policies.eviction_policy import EvictionPolicy


PERMUTATION_CONSTANT: float = 5.0


class ExpertFTPLPolicy(ExpertFTLPolicy):
    """
    Follows the leader, but adds a random initial loss to all experts.
    """

    def __init__(self, capacity: int, policies: [EvictionPolicy]):
        super().__init__(capacity, policies)

    @staticmethod
    def get_name() -> str:
        return "Expert FTPL Policy"

    """
    Gets the expert with the minimum loss, but with some added noise.
    """
    def get_min_loss_expert(self) -> ExpertAdaptedPolicy:
        return min(self.experts, key=lambda e: e.loss + np.random.rand() * PERMUTATION_CONSTANT)
