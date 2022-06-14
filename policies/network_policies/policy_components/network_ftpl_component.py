from policies.policy import Policy


class NetworkFTPLComponent(Policy):

    """
    Updates the cache based on the new request.
    """
    def update(self, request: int) -> None:
        super().update(request)
