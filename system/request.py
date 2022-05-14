class Request:

    catalog_item: int
    current_cost: float

    def __init__(self, catalog_item: int, initial_cost: float = 0):
        self.catalog_item = catalog_item
        self.current_cost = initial_cost

    def get_with_incremented_cost(self, addition: float):
        self.current_cost += addition
        return self
