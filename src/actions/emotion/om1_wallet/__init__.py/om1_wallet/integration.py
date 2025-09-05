
import os

# Загружаем API ключ из переменной окружения
om1_live_6e2f39a3d30c112de7a15fa790e9215e5866e310c7afda8939dbac5ad75eec9797b3d340ec73c393 = os.getenv("OM1_API_KEY")
WALLET_API_KEY = os.getenv("WALLET_API_KEY")

def connect_to_om1():
    """
    Заглушка: функция для подключения к OM1 API
    """
    if not OM1_API_KEY:
        raise ValueError("OM1_API_KEY is not set in environment variables")
    print("Connected to OM1 with API key:", OM1_API_KEY[:5] + "*****")


def process_wallet_payment(amount: float):
    """
    Заглушка: функция для обработки платежа через кошелек
    """
    if not WALLET_API_KEY:
        raise ValueError("WALLET_API_KEY is not set in environment variables")
    print(f"Processing wallet payment of {amount} USD...")
    return {"status": "success", "amount": amount}
