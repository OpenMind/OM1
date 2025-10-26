from om1.core.module_base import InputModuleBase
import random

class EEGInput(InputModuleBase):
    def __init__(self, config):
        super().__init__(config)
        print("EEG input initialized")

    def read(self):
        # Örnek olarak rastgele EEG verisi üretir
        signal = [random.random() for _ in range(8)]
        return {"eeg_signal": signal}
