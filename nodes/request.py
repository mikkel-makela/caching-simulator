class Request:

    catalog_item: int
    current_cost: float = 0

    def __init__(self, catalog_item: int):
        self.catalog_item = catalog_item

    def get_with_incremented_cost(self, addition: float):
        self.current_cost += addition
        return self
