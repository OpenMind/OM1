import logging
import os
import uuid
from dotenv import load_dotenv
from web3 import Web3

from actions.base import ActionConfig, ActionConnector
from actions.pizza_order.interface import PizzaOrderInput


class CreateOrderConnector(ActionConnector[PizzaOrderInput, PizzaOrderInput]):
    """
    Create pizza order, verify payment, check order status.
    """

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        load_dotenv()

        rpc_url = os.getenv("ETH_RPC_URL", "https://eth.llamarpc.com")
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))

        self.pizza_wallet = os.getenv(
            "PIZZA_WALLET_ADDRESS",
            "0xe51B5a66Ace9CCc0A1F381780702d9c3818e8f6F"
        )

        self.orders = []          # saved orders
        self.verified_txs = set() # confirmed transactions

        logging.info("CreateOrderConnector initialized")
        logging.info(f"Connected RPC: {self.w3.is_connected()}")
        logging.info(f"Payment Wallet: {self.pizza_wallet}")

    async def connect(self, io: PizzaOrderInput) -> None:
        try:
            action = (io.action or "").lower()

            if action == "create_order":
                await self._create_order(io)
            elif action == "verify_payment":
                await self._verify_payment(io)
            elif action == "check_order":
                await self._check_order(io)
            else:
                logging.warning(f"Unknown action: {action}")
        except Exception as e:
            logging.error(f"Connector failed: {e}")

    async def _create_order(self, io: PizzaOrderInput) -> None:
        try:
            item = io.item or "pepperoni"
            amount = io.amount or 6
            user_wallet = io.wallet_address or "Not Provided"
            order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"

            logging.info("=== NEW ORDER ===")
            logging.info(f"Item: {item}")
            logging.info(f"Amount: ${amount}")
            logging.info(f"User Wallet: {user_wallet}")
            logging.info("Send USDC payment to:")
            logging.info(self.pizza_wallet)
            logging.info(f"Order ID: {order_id}")
            logging.info("Submit tx hash later with verify_payment.")

            self.orders.append({
                "id": order_id,
                "item": item,
                "amount": amount,
                "wallet": user_wallet,
                "status": "awaiting_payment",
                "tx_hash": None
            })

        except Exception as e:
            logging.error(f"Create order failed: {e}")

    async def _verify_payment(self, io: PizzaOrderInput) -> None:
        try:
            tx = io.payment_hash
            oid = io.order_id

            if not tx or not oid:
                logging.error("order_id and payment_hash required")
                return

            logging.info("=== VERIFY PAYMENT ===")
            logging.info(f"Order: {oid}")
            logging.info(f"Tx: {tx}")

            try:
                receipt = self.w3.eth.get_transaction_receipt(tx)
            except:
                logging.error("Tx not mined or not valid")
                return

            if receipt.get("status", 0) != 1:
                logging.error("Payment failed")
                return

            for o in self.orders:
                if o["id"] == oid:
                    o["status"] = "paid"
                    o["tx_hash"] = tx
                    self.verified_txs.add(tx)

            logging.info("Payment confirmed — order marked paid")

        except Exception as e:
            logging.error(f"Verify payment failed: {e}")

    async def _check_order(self, io: PizzaOrderInput) -> None:
        try:
            oid = io.order_id
            if not oid:
                logging.error("Order ID required")
                return

            order = next((x for x in self.orders if x["id"] == oid), None)

            if not order:
                logging.error(f"Order not found: {oid}")
                return

            logging.info("=== ORDER STATUS ===")
            logging.info(f"Order: {order['id']}")
            logging.info(f"Item: {order['item']}")
            logging.info(f"Amount: ${order['amount']}")
            logging.info(f"Wallet: {order['wallet']}")
            logging.info(f"Status: {order['status']}")
            logging.info(f"Tx Hash: {order['tx_hash']}")

        except Exception as e:
            logging.error(f"Check order failed: {e}")

    def tick(self, output_interface=None):
        return self
