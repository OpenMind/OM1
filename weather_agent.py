from om1 import Agent, run
import requests
import os

class WeatherAgent(Agent):
    """查询天气的智能体示例"""

    def on_message(self, message: str):
        city = message.strip()
        if not city:
            return "请告诉我要查询的城市名称。"

        api_key = os.getenv("OPENWEATHER_API_KEY", "")
        if not api_key:
            return "请先设置环境变量 OPENWEATHER_API_KEY。"

        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&lang=zh_cn&units=metric"
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            return f"🌤 当前 {city} 天气：{desc}，温度 {temp}°C。"
        except Exception as e:
            return f"查询失败：{e}"

if __name__ == "__main__":
    run(WeatherAgent())
