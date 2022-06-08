from system.nodes.node import Node
from system.requests import Request, Response


class MainNode(Node):

    def process_request_and_get_result(self, request: Request) -> Response:
        return request.build_response(was_hit=True)

    def get_hit_ratio(self) -> float:
        return 1.0
