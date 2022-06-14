from policies.ftpl_policy import FTPLPolicy


class Expert:

    policy: FTPLPolicy
    loss: float

    def __init__(self, policy: FTPLPolicy):
        self.policy = policy
        self.loss = 0

    def update_expert(self, file: int):
        if not self.policy.is_present(file):
            self.loss += 1
        self.policy.update(file)
