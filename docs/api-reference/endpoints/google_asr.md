---
title: Google Speech Recognition
description: "Google Speech Recognition (ASR) API Reference"
---

OpenMind proxies the Google Speech Recognition (ASR) API to facilitate speech recognition and transcribe speech to text.

## Usage

The Google ASR API endpoint utilizes WebSockets for efficient, low latency communication.

```bash
wss://api.openmind.org/api/core/google/asr?api_key=<YOUR_API_KEY>
```

The following example demonstrates how to interact with the Google ASR API:

```python
from om1_speech import AudioInputStream

# Initialize the Google ASR API
ws_client = ws.Client(url="wss://api.openmind.org/api/core/google/asr?api_key=<YOUR_API_KEY>")
audio_stream_input = AudioInputStream(audio_data_callback=ws_client.send_message)

# Start the Google ASR API
ws_client.start()
audio_stream_input.start()

# Retrieve the Google ASR API response
ws_client.register_message_callback(lambda msg: print(msg))

while True:
  time.sleep(1)
```

The expected response from the Google ASR API will be in the following format:

```json
{
  "asr_reply": "hello world"
}
```

You can also forward the base64 encoded audio data directly to the API endpoint:

```json
{
  "audio": "base64_encoded_audio_data",
  "rate": 16000
}
```

Note: The `rate` parameter specifies the sample rate of the audio data in Hz. The `rate` parameter is optional and defaults to 16000.
