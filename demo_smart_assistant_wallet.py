from src.integrations.smart_assistant.mock_assistant import MockAssistant
from src.integrations.wallet.coinbase_wallet import CoinbaseWallet
from src.integrations.workflow.order_payment_flow import OrderPaymentFlow

assistant = MockAssistant()
wallet = CoinbaseWallet()

flow = OrderPaymentFlow(assistant, wallet)

result = flow.execute("Coffee", 5.0)
print(result)
