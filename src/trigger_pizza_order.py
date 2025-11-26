#!/usr/bin/env python3
import sys, time, uuid, webbrowser, threading
from flask import Flask, request, jsonify
from flask_cors import CORS

UI_PORT = 8080
WEBHOOK_PORT = 5001
WEBHOOK_ENDPOINT = "/pizzaorder_complete"
DEFAULT_WALLET = "0xe51B5a66Ace9CCc0A1F381780702d9c3818e8f6F"

app = Flask(__name__)
CORS(app)
pizza_data = None

@app.route(WEBHOOK_ENDPOINT, methods=["POST"])
def webhook():
    global pizza_data
    pizza_data = request.json
    print("\n🍕 WEBHOOK RECEIVED")
    print(f"Order ID: {pizza_data.get('order_id')}")
    print(f"Item: {pizza_data.get('item')}")
    print(f"Amount: {pizza_data.get('amount')}")
    print(f"Wallet: {pizza_data.get('wallet')}")
    print(f"TX Hash: {pizza_data.get('tx_hash')}")
    return jsonify({"status": "ok"})

def start_webhook_server():
    print(f"\n🔔 Listening for webhook → http://localhost:{WEBHOOK_PORT}{WEBHOOK_ENDPOINT}")
    app.run(host="0.0.0.0", port=WEBHOOK_PORT, debug=False, use_reloader=False)

def open_pizza_ui(order_id, item, amount, wallet):
    url = f"http://localhost:{UI_PORT}/pizza.html?order_id={order_id}&item={item}&amount={amount}&wallet={wallet}"
    print("\n🌍 Opening Pizza UI")
    print(f"URL → {url}")
    webbrowser.open(url)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🍕 OM1 PIZZA ORDER TRIGGER")
    print("="*60)

    amount = sys.argv[1] if len(sys.argv) > 1 else "6"
    item = sys.argv[2] if len(sys.argv) > 2 else "pepperoni"
    order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"

    threading.Thread(target=start_webhook_server, daemon=True).start()
    time.sleep(1)
    open_pizza_ui(order_id, item, amount, DEFAULT_WALLET)

    print(f"\n⏳ Waiting for payment webhook for order: {order_id}")

    while pizza_data is None:
        time.sleep(1)

    print("\n" + "="*60)
    if pizza_data["tx_hash"]:
        print("🎉 PAYMENT CONFIRMATION RECEIVED")
        print("="*60)
        print(f"TX Hash: {pizza_data['tx_hash']}")
        print(f"Etherscan: https://sepolia.etherscan.io/tx/{pizza_data['tx_hash']}\n")
        print("📢 Now run OM1 verify_payment() to mark as PAID")
    else:
        print("❌ PAYMENT FAILED — No transaction hash received")
    print("="*60)
