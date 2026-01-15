import speech_recognition as sr

def start_listening():
    recognizer = sr.Recognizer()
    # Gürültü eşiğini otomatik ayarla
    recognizer.dynamic_energy_threshold = True
    
    with sr.Microphone() as source:
        print("\n" + "="*40)
        print("LOKAL ÇOK DİLLİ TEST BAŞLADI")
        print("Sizi dinliyorum... (TR, EN, FR, ES, CN)")
        print("="*40)
        
        while True:
            try:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=5)
                # Google ASR'ye gönderiyoruz (Dil olarak 'auto' yerine ana dilleri dener)
                # Buradaki liste Google'ın desteklediği dillerdir.
                text = recognizer.recognize_google(audio, language="tr-TR")
                print(f">>> ALGILANAN (Türkçe): {text}")
                
            except sr.UnknownValueError:
                # Ses anlaşılmadığında sessizce devam et
                pass
            except sr.RequestError as e:
                print(f"Hata oluştu: {e}")
            except KeyboardInterrupt:
                print("\nTest durduruldu.")
                break

if __name__ == "__main__":
    start_listening()
