import speech_recognition as sr
import threading
import json
import logging

class ASRProvider:
    def __init__(self, **kwargs):
        self.is_multi = kwargs.get("is_multi_lang", False)
        self.callback = None
        self.running = False
        self.recognizer = sr.Recognizer()
        self.langs = ["tr-TR", "en-US", "fr-FR", "es-ES"]
        logging.info(f"ASRProvider başlatıldı. Çoklu Dil: {self.is_multi}")

    def register_message_callback(self, callback):
        self.callback = callback

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def _listen_loop(self):
        # Mikrofonun sunucuda hata vermemesi için hata kontrolü
        try:
            with sr.Microphone() as source:
                while self.running:
                    try:
                        audio = self.recognizer.listen(source, phrase_time_limit=5)
                        # Çok dilli tarama
                        for lang in self.langs:
                            try:
                                text = self.recognizer.recognize_google(audio, language=lang)
                                if text and self.callback:
                                    self.callback(json.dumps({"asr_reply": text}))
                                    break
                            except: continue
                    except Exception: continue
        except Exception as e:
            logging.error(f"ASR Donanım Hatası (Normaldir, VMI üzerinde mikrofon yok): {e}")

    def stop(self):
        self.running = False
