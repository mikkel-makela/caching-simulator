from typing import List

from policies.expert_policies.expert import Expert
from policies.policy import Policy


class ExpertPolicy(Policy):

    experts: List[Expert]

    def __init__(self, capacity: int, policies: List[Policy]):
        super().__init__(capacity)
        self.experts = list(map(lambda p: Expert(p), policies))

    """
    Updates caches of all experts.
    """
    def update(self, request: int) -> None:
        super().update(request)
        for expert in self.experts:
            expert.update_expert(request)


