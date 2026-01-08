class MockAssistant:
    def place_order(self, item_name: str):
        return {
            "status": "order_sent",
            "item": item_name,
            "assistant": "mock"
        }
