import os
import asyncio
import base64
import logging
import numpy as np
import cv2
from src.providers.vlm_glm_provider import VLMGLMProvider

# Logging
logging.basicConfig(level=logging.WARNING)  # Sadece hataları göster

async def test_direct_api():
    """Doğrudan API'yi test et (video stream olmadan)"""
    
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        print("❌ API anahtarı yok!")
        return
    
    print("🔧 Doğrudan API testi...")
    
    # 1. Test resmi oluştur
    print("🖼️  Test resmi oluşturuluyor...")
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Farklı şekiller çiz
    cv2.putText(img, "TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(img, (20, 20), (180, 180), (0, 255, 0), 2)
    cv2.circle(img, (100, 100), 40, (255, 0, 0), -1)
    
    # Base64'e çevir
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    print(f"✅ Resim oluşturuldu: 200x200, Base64: {len(frame_base64)} karakter")
    
    # 2. Provider oluştur (kamera olmadan)
    print("\n🤖 Provider oluşturuluyor...")
    provider = VLMGLMProvider(
        api_key=api_key,
        camera_index=-1  # Geçersiz kamera indexi
    )
    
    # 3. Custom callback
    responses = []
    
    def save_response(response):
        print(f"\n📨 Yanıt alındı!")
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            responses.append(content)
            print(f"💭 GLM: {content[:100]}...")
    
    provider.register_message_callback(save_response)
    
    # 4. _process_frame metodunu manuel çağır
    print("\n🚀 Frame GLM-4V'ye gönderiliyor...")
    try:
        await provider._process_frame(frame_base64)
        print("✅ Frame başarıyla işlendi")
        
        if responses:
            print(f"\n📊 Toplam {len(responses)} yanıt alındı")
            for i, resp in enumerate(responses, 1):
                print(f"\n{i}. Yanıt:")
                print(f"   {resp}")
        else:
            print("⚠ Hiç yanıt alınamadı")
            
    except Exception as e:
        print(f"❌ Hata: {type(e).__name__}: {e}")

async def test_multiple_frames():
    """Birden fazla test resmi ile test"""
    
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        return
    
    print("\n\n🎬 ÇOKLU FRAME TESTİ")
    print("=" * 50)
    
    provider = VLMGLMProvider(api_key=api_key)
    
    all_responses = []
    
    def collect_responses(response):
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            all_responses.append(content)
    
    provider.register_message_callback(collect_responses)
    
    # 3 farklı test resmi
    test_prompts = [
        "Bu resimde ne görüyorsun?",
        "Resimdeki ana nesneleri tanımla.",
        "Bu resim ne tür bir sahneyi temsil ediyor olabilir?"
    ]
    
    for i in range(3):
        print(f"\n📸 Test {i+1}/3...")
        
        # Farklı bir resim oluştur
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i]
        cv2.circle(img, (75, 75), 50, color, -1)
        cv2.putText(img, f"TEST {i+1}", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Base64'e çevir
        _, buffer = cv2.imencode('.jpg', img)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # _process_frame metodunu modifiye etmek için provider'ı geçici değiştir
        # Bunun yerine doğrudan API'yi çağıralım
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4/")
        
        try:
            response = await client.chat.completions.create(
                model="glm-4v-plus",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": test_prompts[i]},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}},
                        ],
                    }
                ],
                max_tokens=150,
            )
            
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                print(f"   ✅ Yanıt: {content[:80]}...")
                all_responses.append(content)
                
        except Exception as e:
            print(f"   ❌ Hata: {e}")
    
    print(f"\n📊 Toplam {len(all_responses)} başarılı yanıt")
    
    return all_responses

async def main():
    print("🤖 GLM-4V DOĞRUDAN API TESTİ")
    print("=" * 50)
    
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        print("Lütfen API anahtarını ayarlayın:")
        print('export GLM_API_KEY="anahtarınız"')
        return
    
    print(f"🔑 API Anahtarı son 8 karakter: ...{api_key[-8:]}")
    
    try:
        # Test 1: Doğrudan API
        await test_direct_api()
        
        # Test 2: Çoklu frame
        # await test_multiple_frames()
        
    except ImportError as e:
        print(f"❌ Gerekli kütüphane eksik: {e}")
        print("   Kurulum için: pip install openai numpy opencv-python")
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
