import requests

class HomeAssistantClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def place_order(self, item_name: str):
        """
        Demo order action using Home Assistant REST API
        """
        payload = {
            "entity_id": "input_text.last_order",
            "value": item_name,
        }

        response = requests.post(
            f"{self.base_url}/api/services/input_text/set_value",
            headers=self.headers,
            json=payload,
        )

        response.raise_for_status()

        return {
            "status": "order_sent",
            "item": item_name,
        }
