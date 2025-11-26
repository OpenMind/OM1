from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.get("/order")
def order():
    item = request.args.get("item")
    amount = request.args.get("amount")
    subprocess.Popen(["python", "src/trigger_pizza_order.py", item, amount])
    return {"status": "order_started", "item": item, "amount": amount}

app.run(host="0.0.0.0", port=5002)
