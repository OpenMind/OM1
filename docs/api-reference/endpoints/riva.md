---
title: Riva Speech
description: "Riva Speech Recognition (ASR) and Generation (TTS)"
---

The RIVA ASR/TTS endpoint provides an efficient way to access Riva functionality from your robot running OM1.

## ASR Usage

The ASR endpoint utilizes WebSockets for efficient, low-latency communication.

```bash
wss://api-asr.openmind.org?api_key=<YOUR_API_KEY>
```

The following example demonstrates how to interact with the ASR endpoint with the OpenMind `om1_speech` module:

```python
from om1_speech import AudioInputStream

# Initialize the Riva ASR API
ws_client = ws.Client(url="wss://api-asr.openmind.org?api_key=<YOUR_API_KEY>")
audio_stream_input = AudioInputStream(audio_data_callback=ws_client.send_message)

# Start the Riva ASR API
ws_client.start()
audio_stream_input.start()

# Retrieve the Riva ASR API response
ws_client.register_message_callback(lambda msg: print(msg))

while True:
  time.sleep(1)
```

The endpoint will respond in the following format:

```json
{
  "asr_reply": "hello world"
}
```

## TTS Usage

The `https://api.openmind.org/api/core/riva/tts` endpoint generates speech from text using the Riva Text to Speech (TTS) model. The following code snippet demonstrates how to interact with the TTS endpoint with the OpenMind `riva_tts_provider`:

```python
from providers.riva_tts_provider import RivaTTSProvider

tts = RivaTTSProvider(
  url="https://api.openmind.org/api/core/riva/tts",
  api_key=api_key,
)
tts.start()
tts.add_pending_message("Woof Woof I see a rabbit")
