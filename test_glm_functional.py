import os
import asyncio
import base64
from pathlib import Path
from src.providers.vlm_glm_provider import VLMGLMProvider

async def test_glm_functional():
    """GLM Provider'ın gerçek fonksiyonlarını test eder"""
    
    api_key = os.getenv("GLM_API_KEY", "")
    if not api_key:
        print("❌ HATA: GLM_API_KEY environment variable ayarlanmamış!")
        return
    
    print("🚀 GLM Provider başlatılıyor...")
    provider = VLMGLMProvider(api_key=api_key)
    print("✅ GLM Provider başarıyla başlatıldı")
    
    # 1. Provider'ın metodlarını listele
    print("\n📋 Mevcut metodlar:")
    methods = [m for m in dir(provider) if not m.startswith('_')]
    for method in methods[:10]:  # İlk 10 metodu göster
        print(f"  - {method}")
    
    if len(methods) > 10:
        print(f"  ... ve {len(methods) - 10} metod daha")
    
    # 2. Örnek bir resim testi yap
    print("\n🖼️  Resim analizi testi...")
    
    # Test için örnek resim oluştur (basit bir base64 resim)
    try:
        # Küçük bir test resmi oluştur (1x1 pixel siyah)
        test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        # analyze_image metodunu dene (eğer varsa)
        if hasattr(provider, 'analyze_image'):
            print("  analyze_image metodu bulundu, test ediliyor...")
            
            # Basit bir prompt ile test et
            result = await provider.analyze_image(
                image_base64=test_image_base64,
                prompt="Bu resimde ne görüyorsun?",
                max_tokens=50
            )
            print(f"  ✅ Sonuç: {result}")
        else:
            print("  ⚠ analyze_image metodu bulunamadı")
            
        # Veya generate_content metodunu dene
        if hasattr(provider, 'generate_content'):
            print("\n  generate_content metodu bulundu, test ediliyor...")
            messages = [
                {"role": "user", "content": "Merhaba, nasılsın?"}
            ]
            result = await provider.generate_content(messages=messages)
            print(f"  ✅ Sonuç: {result}")
            
    except Exception as e:
        print(f"  ❌ Hata: {type(e).__name__}: {e}")
    
    print("\n✨ Test tamamlandı!")

if __name__ == "__main__":
    asyncio.run(test_glm_functional())
