import asyncio
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from providers.vlm_glm_provider import VLMGLMProvider

async def run_advanced_tests():
    provider = VLMGLMProvider(api_key="sk-test-key") # Test anahtarı
    
    print("🧪 SENARYO 1: Hatalı Anahtar Tepkisi")
    # Yanlış anahtarla sistemin çöküp çökmediğini kontrol ediyoruz
    res = provider.generate_response("Test", "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
    if "error" in res:
        print(f"✅ Sistem hatayı yakaladı: {res['error']}")
    
    print("\n🧪 SENARYO 2: İstek Süresi Ölçümü")
    start_time = time.time()
    # Bağlantı kurmaya çalışırken ne kadar süre harcadığını ölçüyoruz
    provider.generate_response("Hız testi")
    end_time = time.time()
    print(f"⏱️ Toplam trafik süresi: {end_time - start_time:.2f} saniye")

    print("\n🧪 SENARYO 3: OpenRouter Yapı Denetimi")
    # Maintainer'ın istediği URL yapısının bozulup bozulmadığını teyit ediyoruz
    if provider.base_url.startswith("https://openrouter.ai"):
        print("✅ Endpoint hala doğru.")

if __name__ == "__main__":
    asyncio.run(run_advanced_tests())
