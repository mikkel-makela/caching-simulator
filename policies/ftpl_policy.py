import numpy as np

from policies.ftl_policy import FTLPolicy
from policies.policy import Policy


PERMUTATION_CONSTANT: float = 5.0


class FTPLPolicy(FTLPolicy):
    """
    Follows the leader, but adds a random initial loss to all experts.
    """

    def __init__(self, capacity: int, policies: [Policy]):
        initial_losses = np.random.rand(len(policies)) * -PERMUTATION_CONSTANT
        super().__init__(capacity, policies, initial_losses=initial_losses)

    @staticmethod
    def get_name() -> str:
        return "FTPL Policy"
