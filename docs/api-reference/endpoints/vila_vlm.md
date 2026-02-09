---
title: VILA VLM
description: "VILA Vision-Language Model API Reference"
icon: webhook
---

The VILA VLM API provides real-time vision-language model analysis of video streams. This WebSocket-based endpoint enables low-latency streaming of video frames and receiving intelligent visual descriptions and analysis.

**Base URL:** `wss://api-vila.openmind.org`

**Authentication:** Requires an OpenMind API key passed as a query parameter.

## WebSocket Connection

Establish a persistent WebSocket connection for streaming video frames and receiving real-time VLM analysis.

**Endpoint:** `wss://api-vila.openmind.org?api_key=YOUR_API_KEY`

### Connection Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `api_key` | string | Yes | Your OpenMind API key for authentication |

### Connection Example

```python
import asyncio
import websockets

async def connect_to_vlm():
    async with websockets.connect(
        "wss://api-vila.openmind.org?api_key=om1_live_your_api_key"
    ) as websocket:
        # Send and receive messages
        pass

asyncio.run(connect_to_vlm())
```

## Sending Video Frames

### Message Format

Send video frames as JSON messages over the WebSocket connection:

```json
{
  "timestamp": 1234567890.123,
  "frame": "base64_encoded_jpeg_image"
}
```

### Message Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timestamp` | float | Yes | Unix timestamp when the frame was captured |
| `frame` | string | Yes | Base64-encoded JPEG image data |

### Frame Specifications

- **Format:** JPEG (base64-encoded)
- **Recommended Resolution:** 640x480 pixels (configurable)
- **Recommended FPS:** 30 frames per second (configurable)
- **Quality:** JPEG compression quality 70 (default)

## Receiving VLM Analysis

### Response Format

**VLM Analysis Result:**
```json
{
  "vlm_reply": "The most interesting aspect in this series of images is the man's constant motion of speaking and looking in different directions while sitting in front of a laptop."
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `vlm_reply` | string | Vision-language model analysis of the video frames |

## Usage Examples

### Python Example with VideoStream

The `om1_vlm.VideoStream` wrapper simplifies video capture and streaming:

```python
import asyncio
import websockets
import json
from om1_vlm import VideoStream

async def stream_with_vlm():
    """Stream video to VILA VLM using VideoStream wrapper."""
    uri = "wss://api-vila.openmind.org?api_key=om1_live_your_api_key"

    async with websockets.connect(uri) as websocket:
        # Initialize video stream
        vlm = VideoStream(
            frame_callback=lambda frame: asyncio.create_task(websocket.send(frame)),
            fps=30,
            resolution=(640, 480),
            jpeg_quality=70,
            device_index=0  # Default camera
        )

        # Start video stream
        vlm.start()

        # Receive and process VLM responses
        try:
            async for message in websocket:
                data = json.loads(message)
                if "vlm_reply" in data:
                    print(f"VLM Analysis: {data['vlm_reply']}")
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            vlm.stop()

# Run the streaming client
asyncio.run(stream_with_vlm())
```

### VideoStream Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frame_callback` | Callable | None | Callback function to send frames (e.g., `websocket.send`) |
| `fps` | int | 30 | Frames per second to capture |
| `resolution` | Tuple[int, int] | (640, 480) | Video resolution (width, height) |
| `jpeg_quality` | int | 70 | JPEG compression quality (0-100) |
| `device_index` | int | 0 | Camera device index |

### Custom Implementation

For custom video streaming without the VideoStream wrapper:

```python
import asyncio
import websockets
import json
import base64
import cv2
import time

async def stream_video_to_vlm():
    """Stream video frames to VILA VLM."""
    api_key = "om1_live_your_api_key"
    ws_url = f"wss://api-vila.openmind.org?api_key={api_key}"

    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    async with websockets.connect(ws_url) as websocket:
        print("Connected to VILA VLM")

        # Start receiving task
        async def receive_analysis():
            async for message in websocket:
                data = json.loads(message)
                if "vlm_reply" in data:
                    print(f"VLM: {data['vlm_reply']}")

        receive_task = asyncio.create_task(receive_analysis())

        # Stream video frames
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                # Send frame
                message = {
                    "timestamp": time.time(),
                    "frame": frame_base64
                }
                await websocket.send(json.dumps(message))

                # Maintain 30 FPS
                await asyncio.sleep(1/30)

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            cap.release()
            receive_task.cancel()

# Run the streaming client
asyncio.run(stream_video_to_vlm())
```

### JavaScript/Node.js Example

```javascript
const WebSocket = require('ws');
const { createCanvas, loadImage } = require('canvas');

const API_KEY = 'om1_live_your_api_key';
const WS_URL = `wss://api-vila.openmind.org?api_key=${API_KEY}`;

// Connect to WebSocket
const ws = new WebSocket(WS_URL);

ws.on('open', () => {
    console.log('Connected to VILA VLM');

    // Start streaming frames (example with canvas)
    setInterval(async () => {
        try {
            // Capture or load frame (this is a placeholder)
            const canvas = createCanvas(640, 480);
            const ctx = canvas.getContext('2d');

            // Draw your video frame to canvas here
            // ctx.drawImage(videoFrame, 0, 0);

            // Convert to JPEG base64
            const jpegBuffer = canvas.toBuffer('image/jpeg', { quality: 0.7 });
            const frameBase64 = jpegBuffer.toString('base64');

            // Send frame
            ws.send(JSON.stringify({
                timestamp: Date.now() / 1000,
                frame: frameBase64
            }));
        } catch (error) {
            console.error('Error sending frame:', error);
        }
    }, 1000 / 30); // 30 FPS
});

ws.on('message', (data) => {
    const response = JSON.parse(data);

    if (response.vlm_reply) {
        console.log(`VLM Analysis: ${response.vlm_reply}`);
    }
});

ws.on('error', (error) => {
    console.error('WebSocket error:', error);
});

ws.on('close', () => {
    console.log('Disconnected from VILA VLM');
});
```

## Best Practices

### Video Quality

- Use recommended resolution of 640x480 for optimal balance of quality and bandwidth
- Maintain JPEG quality around 70 for efficient compression
- Ensure good lighting for better visual analysis
- Keep camera stable for consistent results

### Network Optimization

- Send frames at consistent intervals (30 FPS recommended)
- Monitor WebSocket connection health
- Implement reconnection logic for network interruptions
- Buffer frames locally during temporary connection issues

### Performance Tips

- Don't accumulate frames before sending - stream in real-time
- Process VLM responses asynchronously
- Adjust FPS based on network conditions
- Use appropriate resolution for your use case

### Security

- Never hardcode API keys in client-side code
- Use environment variables for API key storage
- Rotate API keys regularly
- Monitor API key usage for suspicious activity

## Error Handling

### Connection Issues

- Verify API key is valid and active
- Check WebSocket support in your environment
- Ensure network allows WebSocket connections
- Test connection with basic example first

### Poor Analysis Quality

- Increase video resolution if bandwidth allows
- Improve lighting conditions
- Reduce motion blur by adjusting camera settings
- Ensure frames are not corrupted during encoding

### Cleanup

Always properly close connections and release resources:

```python
try:
    async for message in websocket:
        # Process messages
        pass
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    vlm.stop()
    # WebSocket context manager handles cleanup automatically
```

<Note>OpenMind developed [om1_modules](https://github.com/OpenMind/OM1-modules) to simplify integration with VILA VLM and other services. For more details, visit [Our GitHub](https://github.com/OpenMind/OM1-modules).</Note>
