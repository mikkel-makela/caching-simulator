from statistics import mode

from policies.expert_policy import ExpertPolicy
from policies.policy import Policy


class FTLPolicy(ExpertPolicy):
    """
    Follows the expert that has the smallest loss.
    """

    def __init__(self, capacity: int, policies: [Policy]):
        super().__init__(capacity, policies)

    @staticmethod
    def get_name() -> str:
        return "FTL Policy"

    """
    Selects the victim from expert advice, chooses the expert with the smallest loss.
    """
    def get_victim(self) -> int:
        min_loss = min(self.experts, key=lambda e: e.loss).loss
        leaders = list(filter(lambda e: e.loss == min_loss, self.experts))
        return mode(list(map(lambda l: l.get_eviction_advice(), leaders)))

    """
    Records losses based on whether the request can be served.
    """
    def record_losses(self, request: int) -> None:
        for expert in self.experts:
            if not expert.can_virtual_cache_serve_request(request):
                expert.loss += 1
