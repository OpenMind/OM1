---
title: Riva Speech
description: "Riva Speech Recognition (ASR) and Text-to-Speech (TTS)"
---

# Riva Speech

The RIVA module provides efficient Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) capabilities powered by NVIDIA Riva for your robot running OM1.

## Overview

OpenMind integrates the NVIDIA Riva's state-of-the-art speech AI models to offer:

- **ASR (Automatic Speech Recognition)**: Real-time speech-to-text conversion with automatic punctuation, profanity filtering, and multi-language support
- **TTS (Text-to-Speech)**: High-quality speech synthesis with customizable voices and languages
- **WebSocket Integration**: Efficient streaming communication for low-latency processing
- **Flexible Audio Input**: Support for microphone, audio streams, and remote audio sources

## ASR Usage

### Cloud-Based ASR (OpenMind API)

The ASR endpoint utilizes WebSockets for efficient, low-latency communication with the OpenMind cloud service.

#### Connection Endpoint

```bash
wss://api-asr.openmind.org?api_key=<YOUR_API_KEY>
```

#### Basic Example

The following example demonstrates how to interact with the ASR endpoint using plain Python:

```python
import asyncio
import websockets
import json
import base64
import pyaudio

async def stream_audio_to_asr():
    """Stream audio to ASR endpoint."""
    uri = "wss://api-asr.openmind.org?api_key=<YOUR_API_KEY>"

    # Audio configuration
    RATE = 16000
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    async with websockets.connect(uri) as websocket:
        print("Connected to ASR service")

        # Send first message with configuration
        first_audio = stream.read(CHUNK)
        first_message = {
            "audio": base64.b64encode(first_audio).decode('utf-8'),
            "rate": RATE,
            "language_code": "en-US"
        }
        await websocket.send(json.dumps(first_message))

        # Start receiving task
        async def receive_transcriptions():
            async for message in websocket:
                data = json.loads(message)
                if "asr_reply" in data:
                    print(f"Recognized: {data['asr_reply']}")

        receive_task = asyncio.create_task(receive_transcriptions())

        # Stream audio
        try:
            while True:
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                message = {
                    "audio": base64.b64encode(audio_data).decode('utf-8')
                }
                await websocket.send(json.dumps(message))
                await asyncio.sleep(0.01)  # Small delay
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            receive_task.cancel()

# Run the streaming client
asyncio.run(stream_audio_to_asr())
```

#### Response Format

The endpoint responds with transcriptions in the following JSON format:

```json
{
  "asr_reply": "hello world"
}
```

### Audio Input Configuration

Configure audio capture using PyAudio:

```python
import pyaudio

# Audio configuration parameters
RATE = 16000                  # Sample rate in Hz
CHUNK = 1024                  # Chunk size in frames
FORMAT = pyaudio.paInt16      # Audio format (16-bit PCM)
CHANNELS = 1                  # Mono audio
DEVICE_INDEX = None           # Use default device (or specify index)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# List available devices
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    print(f"Device {i}: {info['name']}")

# Open audio stream
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK
)
```

## TTS Usage

### Cloud-Based TTS (OpenMind API)

The TTS endpoint generates speech from text using the Riva Text-to-Speech model.

#### Endpoint

```
POST https://api.openmind.org/api/core/riva/tts
```

#### Basic Example

```python
import requests
import os

# API configuration
api_url = "https://api.openmind.org/api/core/riva/tts"
api_key = os.getenv("OPENMIND_API_KEY")

# Request payload
payload = {
    "text": "Hello from OpenMind!",
    "voice": "English-US.Female-1",
    "language_code": "en-US"
}

# Make request
response = requests.post(
    api_url,
    json=payload,
    headers={"Authorization": f"Bearer {api_key}"}
)

if response.status_code == 200:
    # Response contains base64 encoded audio
    audio_data = response.json()["audio"]
    print(f"Generated audio (base64): {audio_data[:50]}...")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### TTS Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | string | Text to convert to speech |
| `voice` | string | Voice identifier (e.g., "English-US.Female-1") |
| `language_code` | string | Language code (e.g., "en-US", "es-ES") |

## Error Handling

### Common Issues

1. **WebSocket connection failed**
   ```
   ERROR: Failed to connect to WebSocket endpoint
   ```
   Solution: Verify API key is valid and check network connectivity

2. **Invalid API key**
   ```
   ERROR: Authentication failed
   ```
   Solution: Ensure you're using a valid OpenMind API key

3. **Audio device not found**
   ```
   ERROR: Failed to open audio device
   ```
   Solution: Check that your microphone is connected and permissions are granted

## Performance Optimization

### Chunk Size Tuning

Optimize chunk size for your use case:

```python
# Lower latency (smaller chunks)
CHUNK = 800  # ~50ms at 16kHz

# Better throughput (larger chunks)
CHUNK = 1600  # ~100ms at 16kHz
```

### Sample Rate Selection

Choose appropriate sample rate based on quality requirements:

- **16 kHz**: Standard telephony quality, lower bandwidth (recommended for ASR)
- **44.1 kHz**: CD quality audio
- **48 kHz**: Professional audio quality

## Security Considerations

### API Key Management

Never hardcode API keys in your source code:

```python
import os
import asyncio
import websockets

async def connect_with_api_key():
    api_key = os.getenv("OPENMIND_API_KEY")
    uri = f"wss://api-asr.openmind.org?api_key={api_key}"

    async with websockets.connect(uri) as websocket:
        # Your application logic here
        pass

asyncio.run(connect_with_api_key())
```

### Best Practices

- Store API keys in environment variables
- Rotate API keys regularly
- Monitor API usage for suspicious activity
- Use HTTPS/WSS for all API communications

## Troubleshooting

### Enable Debug Logging

```python
import logging

# Enable debug logging for your application
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
```

### Check Audio Device

List available audio devices:

```python
import pyaudio

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"Device {i}: {info['name']}")
p.terminate()
```

<Note>OpenMind developed [om1_modules](https://github.com/OpenMind/OM1-modules) to simplify integration with VILA VLM and other services. For more details, visit [Our GitHub](https://github.com/OpenMind/OM1-modules).</Note>
