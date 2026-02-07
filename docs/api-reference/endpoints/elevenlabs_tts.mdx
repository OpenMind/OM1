---
title: ElevenLabs TTS
description: "ElevenLabs Text to Speech (TTS)"
---

The ElevenLabs TTS API converts text into natural-sounding speech using ElevenLabs' advanced text-to-speech models. This endpoint provides high-quality voice synthesis with customizable voice selection, speech speed, and output formats.

**Base URL:** `https://api.openmind.org`

**Authentication:** OpenMind API key is required. Include the key in the `x-api-key` or `Authorization` header.

## Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/elevenlabs/tts` | Generate speech from text using ElevenLabs TTS |

## Generate Speech

Convert text to speech using the ElevenLabs TTS engine with customizable voice and output options.

**Endpoint:** `POST /elevenlabs/tts`

### Request

```bash
curl -X POST https://api.openmind.org/elevenlabs/tts \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "text": "Hello, this is a test of the ElevenLabs text to speech API."
  }'
```

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | Yes | - | The text to convert to speech |
| `voice_id` | string | No | `JBFqnCBsd6RMkjVDRZzb` | ElevenLabs voice ID for the desired voice |
| `model_id` | string | No | `eleven_flash_v2_5` | ElevenLabs model ID to use for synthesis |
| `output_format` | string | No | `mp3_44100_128` | Audio output format specification |
| `speed` | float | No | `1.0` | Speech speed multiplier (0.5 - 2.0) |
| `elevenlabs_api_key` | string | No | - | Optional ElevenLabs API key override |

### Response

**Success (200 OK):**

```json
{
  "response": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tQAAAAAAAAAAAA...",
  "format": "mp3_44100_128"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `response` | string | Base64-encoded audio data ready for decoding and playback |
| `format` | string | Audio format of the returned data (e.g., "mp3_44100_128") |

**Error Responses:**

```json
// 400 Bad Request - Missing or invalid input
{
  "error": "Missing or invalid JSON in request"
}

// 503 Service Unavailable - API key not configured
{
  "error": "ElevenLabs API key not configured"
}

// 503 Service Unavailable - Connection failure
{
  "error": "Failed to connect to ElevenLabs server"
}

// 500 Internal Server Error
{
  "error": "Failed to read response"
}
```

<Note>The returned audio is base64-encoded. You must decode it before playback or saving to a file.</Note>

## Usage Examples

### Basic Text-to-Speech

Convert simple text to speech using default settings:

```bash
curl -X POST https://api.openmind.org/elevenlabs/tts \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "text": "Welcome to OpenMind AGI. This is a demonstration of text to speech conversion."
  }'
```

### Custom Voice and Speed

Use a specific voice with faster speech rate:

```bash
curl -X POST https://api.openmind.org/elevenlabs/tts \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "text": "This speech is faster than normal and uses a custom voice.",
    "voice_id": "JBFqnCBsd6RMkjVDRZzb",
    "speed": 1.3
  }'
```

### Full Configuration

Customize all available parameters:

```bash
curl -X POST https://api.openmind.org/elevenlabs/tts \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "text": "Fully customized text to speech with all parameters specified.",
    "voice_id": "your_voice_id",
    "model_id": "eleven_flash_v2_5",
    "output_format": "mp3_44100_128",
    "speed": 0.9,
    "elevenlabs_api_key": "your_elevenlabs_api_key"
  }'
```

### Save Audio to File

Generate speech and save directly to an MP3 file:

```bash
curl -X POST https://api.openmind.org/elevenlabs/tts \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "text": "This audio will be saved to a file on your local machine."
  }' | jq -r '.response' | base64 -d > output.mp3
```

### With Environment Variables

Store your configuration in environment variables for easier management:

```bash
# Set environment variables
export TTS_VOICE_ID="JBFqnCBsd6RMkjVDRZzb"
export TTS_SPEED="1.1"

# Use in request
curl -X POST https://api.openmind.org/elevenlabs/tts \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d "{
    \"text\": \"Using environment variables for configuration.\",
    \"voice_id\": \"$TTS_VOICE_ID\",
    \"speed\": $TTS_SPEED
  }"
```

## Voice Configuration

### Default Voice

The default voice ID is `JBFqnCBsd6RMkjVDRZzb`. This voice provides clear, natural-sounding English speech suitable for most applications.

### Custom Voices

You can use any ElevenLabs voice ID by specifying it in the `voice_id` parameter. Visit the [ElevenLabs Voice Library](https://elevenlabs.io/voice-library) to explore available voices.

### Speed Control

The `speed` parameter accepts values between 0.5 (half speed) and 2.0 (double speed):
- `0.5` - 50% slower (more deliberate)
- `1.0` - Normal speed (default)
- `1.5` - 50% faster
- `2.0` - Double speed (maximum)

## Output Formats

The default output format is `mp3_44100_128`, which provides high-quality audio at a reasonable file size. The format string indicates:
- Codec: MP3
- Sample Rate: 44,100 Hz
- Bitrate: 128 kbps

Other formats may be supported depending on your ElevenLabs API configuration. Consult the ElevenLabs documentation for available format options.

## Error Handling

All endpoints follow consistent error response patterns:

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Audio generated successfully |
| 400 | Bad Request - Missing required fields or invalid JSON |
| 503 | Service Unavailable - ElevenLabs API unavailable or not configured |
| 500 | Internal Server Error - Server-side processing error |

### Error Response Format

```json
{
  "error": "Descriptive error message"
}
```

### Common Error Scenarios

**Missing Text Field:**
```bash
# This will fail - text is required
curl -X POST https://api.openmind.org/elevenlabs/tts \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{}'

# Response: {"error": "Missing or invalid JSON in request"}
```

**API Key Not Configured:**
If the server-side ElevenLabs API key is not configured and you don't provide one in the request, you'll receive:
```json
{
  "error": "ElevenLabs API key not configured"
}
```

**Connection Issues:**
If the service cannot reach the ElevenLabs API:
```json
{
  "error": "Failed to connect to ElevenLabs server",
  "details": "additional error information"
}
```

## Best Practices

### Audio Decoding

The API returns base64-encoded audio data. Always decode it before use:

```bash
# Decode and save to file
echo "SUQzBAAAAAAAI1RTU0UAAAA..." | base64 -d > audio.mp3

# Or use jq to extract from JSON response
curl ... | jq -r '.response' | base64 -d > audio.mp3
```

<Note>
Note the following best practices when using the ElevenLabs TTS API:
- Audio responses are base64-encoded and must be decoded before playback
- The ElevenLabs API key can be configured server-side or provided per-request
- Default voice and model settings are optimized for English speech
- Large text inputs may take longer to process
</Note>
