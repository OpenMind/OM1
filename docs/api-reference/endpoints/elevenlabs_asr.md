---
title: ElevenLabs ASR
description: "ElevenLabs Automatic Speech Recognition (ASR) API Reference"
icon: webhook
---

The ElevenLabs ASR API provides real-time speech-to-text transcription using ElevenLabs' Scribe v2 model. This WebSocket-based endpoint enables low-latency streaming recognition for live audio processing with voice activity detection and partial transcript delivery.

**Base URL:** `wss://api.openmind.com`

**Authentication:** Requires an OpenMind API key passed as a query parameter.

## Endpoint Overview

| Protocol | Endpoint | Description |
|----------|----------|-------------|
| WebSocket | `/api/core/elevenlabs/asr` | Real-time speech recognition with ElevenLabs Scribe v2 |

> **Note:** The endpoint also supports the `/api/core/v1/` prefix for API versioning (e.g., `/api/core/v1/elevenlabs/asr`).

## WebSocket Connection

Establish a persistent WebSocket connection for streaming audio data and receiving real-time transcription results.

**Endpoint:** `wss://api.openmind.com/api/core/elevenlabs/asr?api_key=YOUR_API_KEY`

### Connection Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `api_key` | string | Yes | Your OpenMind API key for authentication |

### Connection Example

```bash
# Using wscat (install with: npm install -g wscat)
wscat -c "wss://api.openmind.com/api/core/elevenlabs/asr?api_key=om1_live_your_api_key"
```

### Connection Response

Upon successful connection, you'll receive a confirmation message:

```json
{
  "type": "connection",
  "message": "Connected to ElevenLabs ASR service",
  "clientId": "1738713600000-a1b2c3d4e5f6g7h8"
}
```

### Connection Errors

**401 Unauthorized - Missing API Key:**
```json
{
  "error": "Missing API key. Please connect with ?api_key=YOUR_API_KEY"
}
```

**401 Unauthorized - Invalid API Key:**
```json
{
  "error": "Invalid API key: [error details]"
}
```

## Sending Audio Data

### Message Format

Send audio data as JSON messages over the WebSocket connection:

```json
{
  "audio": "base64_encoded_audio_data",
  "rate": 16000,
  "language_code": "en"
}
```

### Message Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `audio` | string | Yes | - | Base64-encoded raw PCM audio data |
| `rate` | integer | No | `16000` | Audio sample rate in Hz. Supported values: `8000`, `16000`, `22050`, `44100` |
| `language_code` | string | No | `"auto"` | BCP-47 language code (e.g., `"en"`, `"es"`, `"fr"`). Use `"auto"` or omit for automatic language detection |

> **Note:** The `rate` and `language_code` parameters only need to be sent with the first message. Subsequent messages can contain only the `audio` field.

### Audio Format Mapping

The sample rate determines the PCM format sent to ElevenLabs:

| Sample Rate (Hz) | PCM Format |
|-----------------|------------|
| 8000 | `pcm_8000` |
| 16000 | `pcm_16000` (default) |
| 22050 | `pcm_22050` |
| 44100 | `pcm_44100` |

> **Recommendation:** Use 16000 Hz for the best balance of quality and bandwidth.

## Receiving Transcription Results

The service delivers two types of transcription events as results become available.

### Partial Transcript

Intermediate, in-progress transcription result emitted as the user speaks:

```json
{
  "type": "partial",
  "asr_reply": "hello wor",
  "clientId": "1738713600000-a1b2c3d4e5f6g7h8",
  "time": 1738713600123
}
```

### Committed Transcript

Final, committed transcription result for a completed utterance:

```json
{
  "asr_reply": "hello world",
  "clientId": "1738713600000-a1b2c3d4e5f6g7h8",
  "time": 1738713600456
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `asr_reply` | string | Transcription text (partial or final) |
| `clientId` | string | Unique identifier for the WebSocket session |
| `type` | string | Message type: `"connection"`, `"partial"`, `"error"`, `"info"` (absent on committed transcripts) |
| `message` | string | Human-readable message for connection, info, or error events |
| `time` | integer | Unix timestamp in milliseconds when the result was produced |

### Info Messages

The server may send informational messages during operation, such as when a recognition session is automatically restarted:

```json
{
  "type": "info",
  "message": "Recognition session restarted",
  "clientId": "1738713600000-a1b2c3d4e5f6g7h8"
}
```

### Error Messages

```json
{
  "type": "error",
  "message": "Speech recognition error: [details]",
  "clientId": "1738713600000-a1b2c3d4e5f6g7h8"
}
```

## Session Limits

| Limit | Value | Description |
|-------|-------|-------------|
| Max streaming duration | 5 minutes | Each internal ElevenLabs session is capped at 5 minutes; the session automatically restarts |
| Silence timeout | Configurable | The connection closes after a period of silence with no detected speech |

When the 5-minute streaming limit is reached, the session is seamlessly restarted and an `"info"` message is sent to the client. Non-recoverable errors will close the WebSocket.

## Audio Specifications

### Supported Audio Format

- **Encoding:** Raw PCM (signed 16-bit little-endian)
- **Sample Rate:** 8000, 16000 (recommended), 22050, or 44100 Hz
- **Channels:** Mono (1 channel)

### Calculating Audio Length

Audio duration is calculated as:

$$\text{duration (s)} = \frac{\text{audio bytes}}{\text{sample rate} \times 2 \times 1}$$

For 16000 Hz mono 16-bit PCM:

$$\text{duration (s)} = \frac{\text{audio bytes}}{32000}$$

## Usage Examples

### Python Example

```python
import asyncio
import websockets
import base64
import json
import pyaudio

API_KEY = "om1_live_your_api_key"
WS_URL = f"wss://api.openmind.com/api/core/elevenlabs/asr?api_key={API_KEY}"

RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

async def stream_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    async with websockets.connect(WS_URL) as websocket:
        connection_msg = await websocket.recv()
        print(f"Connected: {connection_msg}")

        first_audio = stream.read(CHUNK)
        await websocket.send(json.dumps({
            "audio": base64.b64encode(first_audio).decode("utf-8"),
            "rate": RATE,
            "language_code": "en"
        }))

        async def receive():
            async for message in websocket:
                data = json.loads(message)
                if data.get("type") == "partial":
                    print(f"[partial] {data.get('asr_reply', '')}", end="\r")
                elif "asr_reply" in data and data.get("type") != "partial":
                    print(f"\n[final]   {data['asr_reply']}")
                elif data.get("type") == "error":
                    print(f"\n[error]   {data.get('message')}")

        async def send():
            while True:
                chunk = stream.read(CHUNK, exception_on_overflow=False)
                await websocket.send(json.dumps({
                    "audio": base64.b64encode(chunk).decode("utf-8")
                }))

        await asyncio.gather(receive(), send())

asyncio.run(stream_audio())
```

### JavaScript/Node.js Example

```javascript
const WebSocket = require('ws');
const fs = require('fs');

const API_KEY = 'om1_live_your_api_key';
const WS_URL = `wss://api.openmind.com/api/core/elevenlabs/asr?api_key=${API_KEY}`;

const ws = new WebSocket(WS_URL);

ws.on('open', () => {
    console.log('Connected to ElevenLabs ASR');

    const audioFile = fs.readFileSync('audio.raw'); // raw 16-bit PCM
    const chunkSize = 4096;
    let offset = 0;

    ws.send(JSON.stringify({
        audio: audioFile.slice(0, chunkSize).toString('base64'),
        rate: 16000,
        language_code: 'en'
    }));
    offset += chunkSize;

    const interval = setInterval(() => {
        if (offset >= audioFile.length) {
            clearInterval(interval);
            return;
        }
        ws.send(JSON.stringify({
            audio: audioFile.slice(offset, offset + chunkSize).toString('base64')
        }));
        offset += chunkSize;
    }, 100);
});

ws.on('message', (data) => {
    const response = JSON.parse(data);
    if (response.type === 'connection') {
        console.log(`Client ID: ${response.clientId}`);
    } else if (response.type === 'partial') {
        process.stdout.write(`\r[partial] ${response.asr_reply}`);
    } else if (response.asr_reply) {
        console.log(`\n[final]   ${response.asr_reply}`);
    } else if (response.type === 'error') {
        console.error(`[error]   ${response.message}`);
    }
});

ws.on('error', (err) => console.error('WebSocket error:', err));
ws.on('close', () => console.log('Disconnected'));
```

### Using wscat (Command Line)

```bash
# Install wscat
npm install -g wscat

# Connect
wscat -c "wss://api.openmind.com/api/core/elevenlabs/asr?api_key=om1_live_your_api_key"

# Send first message (paste after connection is established)
{"audio":"<BASE64_PCM_DATA>","rate":16000,"language_code":"en"}
```

### Recording Audio for Testing

**Using SoX:**
```bash
# macOS: brew install sox
# Ubuntu: sudo apt-get install sox

# Record raw PCM at 16 kHz mono
sox -d -r 16000 -c 1 -b 16 -e signed-integer -t raw audio.raw
```

**Using FFmpeg:**
```bash
# Convert existing audio file to correct format
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f s16le audio.raw

# Record from microphone (macOS)
ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -f s16le audio.raw
```

## Language Support

Pass a BCP-47 language code in the first message to pin recognition to a specific language. Omit the field or use `"auto"` to let ElevenLabs detect the language automatically.

| Language | Code |
|----------|------|
| English | `en` |
| Spanish | `es` |
| French | `fr` |
| German | `de` |
| Italian | `it` |
| Portuguese | `pt` |
| Japanese | `ja` |
| Korean | `ko` |
| Chinese (Mandarin) | `zh` |
| Dutch | `nl` |
| Polish | `pl` |
| Russian | `ru` |

> **Note:** For a complete list of supported languages, refer to the [ElevenLabs Speech-to-Text documentation](https://elevenlabs.io/docs/speech-to-text).
