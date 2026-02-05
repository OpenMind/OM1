---
title: Google Speech Recognition
description: "Google Speech Recognition (ASR) API Reference"
---

The Google ASR API provides real-time speech-to-text transcription using Google Cloud Speech-to-Text. This WebSocket-based endpoint enables low-latency streaming recognition for live audio processing.

**Base URL:** `wss://api.openmind.org`

**Authentication:** Requires an OpenMind API key passed as a query parameter.

## Endpoints Overview

| Protocol | Endpoint | Description |
|----------|----------|-------------|
| WebSocket | `/api/core/google/asr` | Real-time speech recognition via WebSocket connection |

## WebSocket Connection

Establish a persistent WebSocket connection for streaming audio data and receiving real-time transcription results.

**Endpoint:** `wss://api.openmind.org/api/core/google/asr?api_key=YOUR_API_KEY`

### Connection Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `api_key` | string | Yes | Your OpenMind API key for authentication |

### Connection Example

```bash
# Using wscat (install with: npm install -g wscat)
wscat -c "wss://api.openmind.org/api/core/google/asr?api_key=om1_live_your_api_key"
```

### Connection Response

Upon successful connection, you'll receive a confirmation message:

```json
{
  "type": "connection",
  "message": "Connected to ASR service",
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
  "language_code": "en-US"
}
```

### Message Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `audio` | string | Yes | - | Base64-encoded audio data (LINEAR16 format) |
| `rate` | integer | No | `16000` | Audio sample rate in Hz |
| `language_code` | string | No | `"en-US"` | Language code for recognition (e.g., "en-US", "es-ES", "fr-FR") |

<Note>
Note the following when sending audio data:
- The `rate` and `language_code` parameters only need to be sent with the first message. Subsequent messages can contain only the `audio` field.
- Audio must be LINEAR16 PCM encoded
- Maximum streaming duration is 4 minutes (240 seconds) per session
</Note>

## Receiving Transcription Results

### Response Format

**Transcription Result:**
```json
{
  "asr_reply": "hello world",
  "clientId": "1738713600000-a1b2c3d4e5f6g7h8"
}
```

**Error Message:**
```json
{
  "type": "error",
  "message": "Error description",
  "clientId": "1738713600000-a1b2c3d4e5f6g7h8"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `asr_reply` | string | Final transcription result for the audio segment |
| `clientId` | string | Unique identifier for the WebSocket session |
| `type` | string | Message type ("connection", "error") |
| `message` | string | Human-readable message for connection or error events |

## Audio Specifications

### Supported Audio Format

- **Encoding:** LINEAR16 (16-bit PCM)
- **Sample Rate:** 16000 Hz (recommended) or custom rate specified in first message
- **Channels:** Mono (1 channel)
- **Sample Width:** 2 bytes (16-bit)

### Calculating Audio Length

Audio duration is calculated as:
```
duration_seconds = audio_bytes / (sample_rate × sample_width × channels)
```

For 16000 Hz mono LINEAR16:
```
duration_seconds = audio_bytes / (16000 × 2 × 1) = audio_bytes / 32000
```

## Usage Examples

### Python Example

```python
import asyncio
import websockets
import base64
import json
import pyaudio

API_KEY = "om1_live_your_api_key"
WS_URL = f"wss://api.openmind.org/api/core/google/asr?api_key={API_KEY}"

# Audio configuration
RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

async def stream_audio():
    """Stream audio from microphone to Google ASR."""
    audio = pyaudio.PyAudio()

    # Open audio stream
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    async with websockets.connect(WS_URL) as websocket:
        # Receive connection confirmation
        connection_msg = await websocket.recv()
        print(f"Connected: {connection_msg}")

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
                    print(f"Transcript: {data['asr_reply']}")
                elif "type" in data and data["type"] == "error":
                    print(f"Error: {data['message']}")

        receive_task = asyncio.create_task(receive_transcriptions())

        # Stream audio
        try:
            while True:
                audio_data = stream.read(CHUNK)
                message = {
                    "audio": base64.b64encode(audio_data).decode('utf-8')
                }
                await websocket.send(json.dumps(message))
                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            receive_task.cancel()

# Run the streaming client
asyncio.run(stream_audio())
```

### JavaScript/Node.js Example

```javascript
const WebSocket = require('ws');
const fs = require('fs');

const API_KEY = 'om1_live_your_api_key';
const WS_URL = `wss://api.openmind.org/api/core/google/asr?api_key=${API_KEY}`;

// Connect to WebSocket
const ws = new WebSocket(WS_URL);

ws.on('open', () => {
    console.log('Connected to Google ASR');

    // Read audio file and send in chunks
    const audioFile = fs.readFileSync('audio.raw'); // LINEAR16 PCM audio
    const chunkSize = 4096;
    let offset = 0;

    // Send first chunk with configuration
    const firstChunk = audioFile.slice(0, chunkSize);
    ws.send(JSON.stringify({
        audio: firstChunk.toString('base64'),
        rate: 16000,
        language_code: 'en-US'
    }));
    offset += chunkSize;

    // Send remaining chunks
    const interval = setInterval(() => {
        if (offset >= audioFile.length) {
            clearInterval(interval);
            return;
        }

        const chunk = audioFile.slice(offset, offset + chunkSize);
        ws.send(JSON.stringify({
            audio: chunk.toString('base64')
        }));
        offset += chunkSize;
    }, 100);
});

ws.on('message', (data) => {
    const response = JSON.parse(data);

    if (response.type === 'connection') {
        console.log(`Client ID: ${response.clientId}`);
    } else if (response.asr_reply) {
        console.log(`Transcript: ${response.asr_reply}`);
    } else if (response.type === 'error') {
        console.error(`Error: ${response.message}`);
    }
});

ws.on('error', (error) => {
    console.error('WebSocket error:', error);
});

ws.on('close', () => {
    console.log('Disconnected from Google ASR');
});
```

### Using wscat (Command Line)

```bash
# Install wscat
npm install -g wscat

# Connect to the WebSocket
wscat -c "wss://api.openmind.org/api/core/google/asr?api_key=om1_live_your_api_key"

# Send a message (paste into the terminal after connection)
{"audio":"UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAAB9AAACABAAZGF0YQAAAAA=","rate":16000,"language_code":"en-US"}
```

### Recording Audio for Testing

**Using SoX (Sound eXchange):**
```bash
# Install SoX
# macOS: brew install sox
# Ubuntu: sudo apt-get install sox

# Record audio in correct format
sox -d -r 16000 -c 1 -b 16 -e signed-integer -t raw audio.raw

# Or record as WAV
sox -d -r 16000 -c 1 -b 16 audio.wav
```

**Using FFmpeg:**
```bash
# Convert existing audio to correct format
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f s16le audio.raw

# Record from microphone
ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -f s16le audio.raw
```

## Language Support

The ASR service supports multiple languages. Specify the language code in the first message:

| Language | Code |
|----------|------|
| English (US) | `en-US` |
| English (UK) | `en-GB` |
| Spanish (Spain) | `es-ES` |
| Spanish (Latin America) | `es-419` |
| French | `fr-FR` |
| German | `de-DE` |
| Italian | `it-IT` |
| Portuguese (Brazil) | `pt-BR` |
| Japanese | `ja-JP` |
| Korean | `ko-KR` |
| Chinese (Mandarin) | `zh-CN` |

<Note>For a complete list of supported languages, refer to the [Google Cloud Speech-to-Text documentation](https://cloud.google.com/speech-to-text/docs/languages).</Note>

## Error Handling

### Common Errors

**Invalid Message Format:**
```json
{
  "type": "error",
  "message": "Invalid message format: [details]",
  "clientId": "1738713600000-a1b2c3d4e5f6g7h8"
}
```

**Missing Audio Field:**
```json
{
  "type": "error",
  "message": "Invalid message format: 'audio' field missing",
  "clientId": "1738713600000-a1b2c3d4e5f6g7h8"
}
```

**Audio Decoding Error:**
```json
{
  "type": "error",
  "message": "Failed to decode audio: [details]",
  "clientId": "1738713600000-a1b2c3d4e5f6g7h8"
}
```

**Speech Recognition Error:**
```json
{
  "type": "error",
  "message": "Speech recognition error: [details]",
  "clientId": "1738713600000-a1b2c3d4e5f6g7h8"
}
```

### Handling Connection Loss

The WebSocket connection may close due to:
- Network interruptions
- 4-minute streaming limit reached
- Client disconnect
- Server errors

Implement reconnection logic in your client:

```python
async def connect_with_retry(max_retries=3):
    for attempt in range(max_retries):
        try:
            async with websockets.connect(WS_URL) as websocket:
                await stream_audio(websocket)
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

## Session Management

### Streaming Limit

Each recognition session has a maximum duration of **4 minutes (240 seconds)**. After this time:
- The current stream will automatically restart
- A new recognition session will begin
- Audio processing continues seamlessly

### Session Cleanup

When the WebSocket connection closes:
- All buffered audio is processed
- Final transcriptions are sent
- Usage tracking is recorded
- Resources are cleaned up

### Client Identification

Each connection receives a unique `clientId` in the format:
```
{timestamp}-{random_hex}
```

This ID is included in all server responses for tracking and debugging purposes.

## Cost Calculation

Speech recognition costs are calculated based on the total audio duration processed:

```
cost_in_omcu = audio_duration_seconds × per_second_rate
```

Usage is tracked and billed to the API key provided in the connection URL.

<Note>
Note the following about cost calculation:
- Audio length is calculated automatically from the data sent
- Only successfully processed audio is billed
- Usage details are available in your OpenMind dashboard
</Note>

## Best Practices

### Audio Quality

- Use high-quality audio input (clear speech, minimal background noise)
- Maintain consistent audio levels
- Use the recommended 16000 Hz sample rate for optimal recognition
- Send audio in consistent chunk sizes (1024-4096 bytes recommended)

### Network Optimization

- Implement exponential backoff for reconnection attempts
- Buffer audio locally during temporary connection issues
- Monitor WebSocket connection health
- Handle network interruptions gracefully

### Error Handling

- Always validate the API key before establishing connections
- Check for error messages in server responses
- Implement retry logic for transient failures
- Log client IDs for debugging and support requests

### Performance Tips

- Send audio chunks at regular intervals (every 50-100ms)
- Avoid sending very large or very small chunks
- Don't accumulate audio before sending - stream in real-time
- Process transcription results asynchronously

### Security

- Never hardcode API keys in client-side code
- Use environment variables for API key storage
- Rotate API keys regularly
- Monitor API key usage for suspicious activity

## Troubleshooting

### No Transcription Results

- Verify audio format is LINEAR16 PCM
- Check sample rate matches the `rate` parameter
- Ensure audio contains clear speech
- Verify language code matches the spoken language

### Connection Issues

- Confirm API key is valid and active
- Check WebSocket support in your environment
- Verify network allows WebSocket connections
- Test connection with wscat first

### Poor Recognition Quality

- Increase audio quality/bitrate
- Reduce background noise
- Speak clearly and at normal pace
- Try adjusting the language model if available

### Buffer Full Warnings

If you see "Audio stream buffer full" in logs:
- Reduce the rate of audio sending
- Increase chunk send interval
- Check for network congestion
- Verify client is reading responses

## Example: Complete Integration

Here's a complete example integrating microphone input, WebSocket streaming, and real-time display:

```python
import asyncio
import websockets
import json
import base64
import pyaudio
from typing import Callable

class GoogleASRClient:
    """Complete Google ASR WebSocket client."""

    def __init__(self, api_key: str, language: str = "en-US"):
        self.api_key = api_key
        self.language = language
        self.ws_url = f"wss://api.openmind.org/api/core/google/asr?api_key={api_key}"
        self.client_id = None

        # Audio config
        self.rate = 16000
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1

        self.transcript_callback = None

    def on_transcript(self, callback: Callable[[str], None]):
        """Register callback for transcription results."""
        self.transcript_callback = callback
        return self

    async def start(self):
        """Start streaming audio and receiving transcriptions."""
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        try:
            async with websockets.connect(self.ws_url) as ws:
                # Handle connection
                conn_msg = json.loads(await ws.recv())
                self.client_id = conn_msg.get('clientId')
                print(f"Connected with ID: {self.client_id}")

                # Send first message with config
                first_audio = stream.read(self.chunk)
                await ws.send(json.dumps({
                    "audio": base64.b64encode(first_audio).decode(),
                    "rate": self.rate,
                    "language_code": self.language
                }))

                # Create tasks for sending and receiving
                send_task = asyncio.create_task(self._send_audio(ws, stream))
                recv_task = asyncio.create_task(self._receive_transcripts(ws))

                # Wait for tasks
                await asyncio.gather(send_task, recv_task)

        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    async def _send_audio(self, ws, stream):
        """Send audio chunks to the WebSocket."""
        try:
            while True:
                audio_data = stream.read(self.chunk, exception_on_overflow=False)
                message = {
                    "audio": base64.b64encode(audio_data).decode()
                }
                await ws.send(json.dumps(message))
                await asyncio.sleep(0.05)  # 50ms between chunks
        except Exception as e:
            print(f"Send error: {e}")

    async def _receive_transcripts(self, ws):
        """Receive and process transcription results."""
        try:
            async for message in ws:
                data = json.loads(message)

                if "asr_reply" in data and self.transcript_callback:
                    self.transcript_callback(data["asr_reply"])
                elif data.get("type") == "error":
                    print(f"Error: {data.get('message')}")
        except Exception as e:
            print(f"Receive error: {e}")

# Usage
async def main():
    client = GoogleASRClient(
        api_key="om1_live_your_api_key",
        language="en-US"
    )

    # Register callback
    client.on_transcript(lambda text: print(f">> {text}"))

    # Start streaming
    await client.start()

if __name__ == "__main__":
    asyncio.run(main())
```

## Additional Resources

- [Google Cloud Speech-to-Text Documentation](https://cloud.google.com/speech-to-text/docs)
- [Supported Languages](https://cloud.google.com/speech-to-text/docs/languages)
- [Audio Encoding Best Practices](https://cloud.google.com/speech-to-text/docs/encoding)
