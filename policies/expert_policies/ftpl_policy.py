from typing import List

import numpy as np

from policies.expert_policies.expert import Expert
from policies.policy import Policy


PERMUTATION_FACTOR: float = 1.0


class ExpertFTPLPolicy(Policy):

    _experts: List[Expert]

    def __init__(self, capacity: int, policies: List[Policy]):
        super().__init__(capacity)
        self._experts = list(map(lambda p: Expert(p), policies))

    @staticmethod
    def get_name() -> str:
        return "Expert FTPL"

    """
    Updates caches of all experts and selects the one with the smallest loss.
    """
    def update(self, request: int) -> None:
        super().update(request)
        for expert in self._experts:
            expert.update_expert(request)

        self.cache = self.get_lowest_loss_expert().policy.cache

    def get_lowest_loss_expert(self) -> Expert:
        losses = np.array(list(map(lambda e: e.loss, self._experts))) + np.random.normal(
            loc=0,
            scale=PERMUTATION_FACTOR,
            size=len(self._experts)
        )
        return self._experts[np.argmin(losses)]
