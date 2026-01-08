class OrderPaymentFlow:
    def __init__(self, assistant_client, wallet_client):
        self.assistant = assistant_client
        self.wallet = wallet_client

    def execute(self, item_name: str, price: float):
        order_result = self.assistant.place_order(item_name)
        payment_result = self.wallet.pay(price)

        confirmation = (
            "Order placed and payment successful"
            if payment_result["status"] == "success"
            else "Payment failed"
        )

        return {
            "order": order_result,
            "payment": payment_result,
            "confirmation": confirmation,
        }
