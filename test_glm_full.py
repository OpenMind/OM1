import os
import asyncio
import logging
import cv2
import numpy as np
from src.providers.vlm_glm_provider import VLMGLMProvider

# Logging'i ayarla
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GLMTestRunner:
    def __init__(self):
        self.api_key = os.getenv("GLM_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GLM_API_KEY ayarlanmamış! 'export GLM_API_KEY=anahtarınız' komutu ile ayarlayın")
        
        print("🚀 GLM Provider başlatılıyor...")
        
        # WebSocket URL'si olmadan başlat (sadece yerel test için)
        self.provider = VLMGLMProvider(
            api_key=self.api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            fps=2,  # Düşük FPS - test için yeterli
            stream_url=None,  # WebSocket yok
            camera_index=0  # Varsayılan kamera
        )
        
        self.response_count = 0
        self.responses = []
    
    def on_glm_response(self, response):
        """GLM'den gelen cevapları işle"""
        self.response_count += 1
        
        print(f"\n{'='*60}")
        print(f"📥 YANIT #{self.response_count}")
        print(f"{'='*60}")
        
        # Response'dan metni çıkar
        try:
            if hasattr(response, 'choices') and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, 'content'):
                    content = message.content
                    print(f"🤖 GLM: {content}")
                    
                    # Response'u kaydet
                    self.responses.append({
                        'timestamp': asyncio.get_event_loop().time(),
                        'content': content
                    })
                else:
                    print(f"📊 Response yapısı: {response}")
            else:
                print(f"📦 Ham response: {response}")
        except Exception as e:
            print(f"⚠ Response işleme hatası: {e}")
            print(f"📦 Response: {response}")
    
    async def run_test(self, duration=30):
        """Testi çalıştır"""
        print(f"\n🎬 TEST BAŞLIYOR - Süre: {duration} saniye")
        print(f"📷 Kamera: {self.provider.video_stream.device_index}")
        print(f"📊 FPS: {self.provider.video_stream.fps}")
        print(f"🔗 Base URL: {self.provider.api_client.base_url}")
        
        # Callback'i kaydet
        self.provider.register_message_callback(self.on_glm_response)
        
        try:
            # Provider'ı başlat
            print("\n▶️  Provider başlatılıyor...")
            self.provider.start()
            print("✅ Provider başlatıldı")
            
            # Süre boyunca bekle
            print(f"\n⏳ {duration} saniye boyunca frame analizi yapılıyor...")
            print("   (Kameranız açık olmalı!)")
            
            for i in range(duration):
                await asyncio.sleep(1)
                remaining = duration - i - 1
                print(f"\r⏱️  Kalan: {remaining:2d}s | Yanıtlar: {self.response_count:2d}", end="", flush=True)
            
            print()  # Yeni satır
            
        except KeyboardInterrupt:
            print("\n\n⚠ Kullanıcı tarafından durduruldu")
        except Exception as e:
            print(f"\n❌ Hata: {e}")
        finally:
            # Provider'ı durdur
            print("\n⏹️  Provider durduruluyor...")
            self.provider.stop()
            print("✅ Provider durduruldu")
            
            # Sonuçları göster
            self.show_results()
    
    def show_results(self):
        """Test sonuçlarını göster"""
        print(f"\n{'='*60}")
        print("📊 TEST SONUÇLARI")
        print(f"{'='*60}")
        print(f"Toplam süre: 30 saniye")
        print(f"Toplam yanıt: {self.response_count}")
        print(f"Ortalama yanıt süresi: {30/self.response_count if self.response_count > 0 else 'N/A':.1f} saniye")
        
        if self.responses:
            print(f"\n📝 ALINAN YANITLAR:")
            for i, resp in enumerate(self.responses, 1):
                print(f"\n{i}. [{resp['timestamp']:.1f}s]:")
                # İlk 200 karakteri göster
                preview = resp['content'][:200] + ("..." if len(resp['content']) > 200 else "")
                print(f"   {preview}")
        
        print(f"\n🎉 Test tamamlandı!")

async def test_with_static_image():
    """Statik resim ile test (kamera yoksa)"""
    print("\n\n🖼️  STATİK RESİM TESTİ")
    print("=" * 60)
    
    # Basit bir test resmi oluştur
    height, width = 100, 100
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Ortasına kırmızı bir kare çiz
    cv2.rectangle(test_image, (20, 20), (80, 80), (0, 0, 255), -1)
    
    # Base64'e çevir
    _, buffer = cv2.imencode('.jpg', test_image)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    print(f"Test resmi oluşturuldu: {width}x{height} pixels")
    print(f"Base64 boyutu: {len(frame_base64)} karakter")
    
    # Manual olarak _process_frame metodunu çağır
    api_key = os.getenv("GLM_API_KEY")
    provider = VLMGLMProvider(api_key=api_key)
    
    # Callback fonksiyonu
    def test_callback(response):
        print(f"\n📨 Callback çağrıldı!")
        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
            print(f"🤖 GLM yanıtı: {content}")
    
    provider.register_message_callback(test_callback)
    
    print("\n🔍 Frame GLM'e gönderiliyor...")
    await provider._process_frame(frame_base64)
    print("✅ Frame işlendi")

async def main():
    print("=" * 70)
    print("🤖 GLM-4V VISION LANGUAGE MODEL TEST SUITE")
    print("=" * 70)
    
    # Anahtar kontrolü
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        print("❌ HATA: GLM_API_KEY environment variable ayarlanmamış!")
        print("   Lütfen şu komutu çalıştırın:")
        print('   export GLM_API_KEY="9c99caefb78c4082a8c340d0f9bab5b1.C8ujmAYlKDvX7vjC"')
        return
    
    print(f"✅ API Anahtarı: {'*' * 20}{api_key[-8:]}")
    
    try:
        # Kamera testini çalıştır
        test_runner = GLMTestRunner()
        await test_runner.run_test(duration=30)
        
    except Exception as e:
        print(f"❌ Test başlatılamadı: {e}")
        print("\n⚠ Alternatif test (statik resim) deneyelim...")
        
        # Global import
        import base64
        
        try:
            await test_with_static_image()
        except Exception as e2:
            print(f"❌ Statik resim testi de başarısız: {e2}")

if __name__ == "__main__":
    asyncio.run(main())
