import asyncio
import logging
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "src"))
from inputs.plugins.google_asr import GoogleASRInput, GoogleASRSensorConfig

async def test_asr():
    logging.basicConfig(level=logging.INFO)
    
    # Burada robotun kendi anahtarını kullanıyoruz. 
    # Eğer hala 401 alırsak, anahtarın ASR yetkisi yok demektir.
    MY_API_KEY = "om1_live_9dec0294cbd624613dab0c2481e0c6d822d78daae8129f19" 
    
    config = GoogleASRSensorConfig(
        language="auto",
        api_key=MY_API_KEY
    )
    
    try:
        asr_input = GoogleASRInput(config)
        print("\n" + "="*40)
        print("BAĞLANTI DENENİYOR...")
        print("="*40)
        
        while True:
            raw_text = await asr_input._poll()
            if raw_text:
                print(f"\n>>> ALGILANAN SES: {raw_text}")
            await asyncio.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nDurduruldu.")

if __name__ == "__main__":
    asyncio.run(test_asr())
