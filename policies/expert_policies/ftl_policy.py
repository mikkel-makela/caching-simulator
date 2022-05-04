from policies.expert_policies.expert_adapter.expert_adapter import ExpertAdaptedPolicy
from policies.expert_policies.expert_policy import ExpertPolicy
from policies.eviction_policy import EvictionPolicy


class ExpertFTLPolicy(ExpertPolicy):
    """
    Follows the expert that has the smallest loss.
    """

    def __init__(self, capacity: int, policies: [EvictionPolicy]):
        super().__init__(capacity, policies)

    @staticmethod
    def get_name() -> str:
        return "Expert FTL Policy"

    """
    Selects the victim from expert advice, chooses the expert with the smallest loss.
    """
    def get_victim(self) -> int:
        return self.get_min_loss_expert().get_eviction_advice()

    """
    Records losses based on whether the request can be served.
    """
    def record_losses(self, request: int) -> None:
        for expert in self.experts:
            if not expert.can_virtual_cache_serve_request(request):
                expert.loss += 1

    """
    Gets the expert with the minimum loss.
    """
    def get_min_loss_expert(self) -> ExpertAdaptedPolicy:
        return min(self.experts, key=lambda e: e.loss)

    """
    Resets the policy and all experts.
    """
    def reset(self) -> None:
        super().reset()
        for expert in self.experts:
            expert.reset()
