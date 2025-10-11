# tools/asr-eval/ws_client.py
import asyncio
import json
import base64
import websockets

class ASRWebsocketClient:
    async def send_audio(self, uri, audio_bytes):
        async with websockets.connect(uri) as ws:
            # Send initial config message (minimal required fields)
            config_msg = json.dumps({
                "languageCode": "en-US",
                "encoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "channels": 1,
                "bitsPerSample": 16
            })
            await ws.send(config_msg)

            # Send audio message
            audio_msg = json.dumps({
                "audio": base64.b64encode(audio_bytes).decode("utf-8")
            })
            await ws.send(audio_msg)

            # Wait for server response
            response = await ws.recv()
            response_obj = json.loads(response)
            return response_obj.get("transcript", "")
