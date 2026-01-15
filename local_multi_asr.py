import speech_recognition as sr
import time

def recognize_multi(recognizer, audio):
    langs = ["tr-TR", "en-US", "fr-FR"]
    for lang in langs:
        try:
            text = recognizer.recognize_google(audio, language=lang)
            return text, lang
        except:
            continue
    return None, None

def start_listening():
    r = sr.Recognizer()
    r.dynamic_energy_threshold = True
    
    try:
        with sr.Microphone() as source:
            print("\n" + "="*40)
            print("SİSTEM HAZIR: 401 HATASI DEVRE DIŞI")
            print("Diller: Türkçe, İngilizce, Fransızca")
            print("="*40)
            print("Lütfen bir şeyler söyleyin...")
            
            while True:
                try:
                    audio = r.listen(source, timeout=None, phrase_time_limit=5)
                    text, lang = recognize_multi(r, audio)
                    
                    if text:
                        print(f"\n[Algılanan Dil: {lang}]")
                        print(f">>> MESAJ: {text}")
                    else:
                        print(".", end="", flush=True)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"\nHata: {e}")
    except Exception as e:
        print(f"Mikrofon Hatası: {e}")
        print("Lütfen 'pip install PyAudio' komutunun başarıyla tamamlandığından emin olun.")

if __name__ == "__main__":
    start_listening()
