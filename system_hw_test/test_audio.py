import sounddevice as sd

samplerates = 8000, 16000, 32000, 44100, 48000, 96000, 128000

all_devices = sd.query_devices()
num_devices = len(all_devices)

for i in range(num_devices):
    supported_samplerates = []
    dev = all_devices[i]
    inc = dev["max_input_channels"]
    outc = dev["max_output_channels"]
    if inc + outc > 4:
        # real devices do not have 128 channels etc
        continue
    for fs in samplerates:
        try:
            sd.check_output_settings(device=i, samplerate=fs)
        except Exception:
            pass
        else:
            supported_samplerates.append(fs)
    print(f"{dev['name']} rates: {supported_samplerates} in:{inc} out:{outc}")
