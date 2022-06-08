from __future__ import annotations

from dataclasses import dataclass


class Request:

    catalog_item: int
    cost: float

    def __init__(self, catalog_item: int, initial_cost: float = 0):
        self.catalog_item = catalog_item
        self.cost = initial_cost

    def get_with_incremented_cost(self, addition: float):
        self.cost += addition
        return self

    def build_response(self, was_hit: bool) -> Response:
        return Response(self.cost, was_hit)


@dataclass
class Response:
    cost: float
    was_hit: bool
