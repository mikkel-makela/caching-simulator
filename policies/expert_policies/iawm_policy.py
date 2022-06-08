import random
from typing import List

import numpy as np

from policies.expert_policies.expert_policy import ExpertPolicy
from policies.policy import Policy


def get_optimal_last_loss(previous_losses: np.ndarray) -> float:
    """
    Gets the best expert to follow with hindsight, returns their loss.
    :return: Optimal last loss
    """
    return np.min(previous_losses)


class IAWMPolicy(ExpertPolicy):

    weights: np.ndarray

    def __init__(self, capacity: int, policies: List[Policy]):
        super().__init__(capacity, policies)
        self.weights = np.zeros(len(policies))

    @staticmethod
    def get_name() -> str:
        return "IAWM+FTPL"

    """
    Updates caches of all experts and selects the one with the largest weight.
    """
    def update(self, request: int) -> None:
        previous_losses = np.array(list(map(lambda e: e.loss, self.experts)))
        super().update(request)
        self.weights = self.get_updated_weights(previous_losses)
        chosen_expert = random.choices(self.experts, weights=self.weights)[0]
        self.cache = chosen_expert.policy.cache

    def get_updated_weights(self, previous_losses: np.ndarray) -> np.ndarray:
        optimal_last_loss = np.min(previous_losses)
        e_t = 0.25 \
            if optimal_last_loss == 0 \
            else min([0.25, np.sqrt((2 * np.log(self.weights.size) / optimal_last_loss))])
        a_t = 1 / (1 - e_t)
        w_t = sum([
            np.power(a_t, -previous_losses[i])
            for i in range(len(self.experts))
        ])

        return np.array(
            [np.power(a_t, -previous_losses[i]) / w_t for i in range(self.weights.size)]
        )
