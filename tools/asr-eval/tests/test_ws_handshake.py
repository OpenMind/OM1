# tools/asr-eval/tests/test_ws_handshake.py
import asyncio
import base64
import json
import pytest
import websockets

# We will import the client later. For now this test expects the helper to exist.
# On the base commit this import will fail (good — test should fail initially).
from tools.asr_eval import ws_client

@pytest.mark.asyncio
async def test_initial_config_and_audio_frames():
    async def server_handler(ws, path):
        msg = await ws.recv()
        obj = json.loads(msg)
        required = {"languageCode", "encoding", "sampleRateHertz", "channels", "bitsPerSample"}
        assert required.issubset(set(obj.keys()))
        audio_msg = await ws.recv()
        audio_obj = json.loads(audio_msg)
        assert "audio" in audio_obj
        audio_bytes = base64.b64decode(audio_obj["audio"])
        assert len(audio_bytes) > 0
        await ws.send(json.dumps({"type": "final", "transcript": "hello world"}))

    server = await websockets.serve(server_handler, "localhost", 8765)
    await asyncio.sleep(0.01)

    try:
        transcript = await ws_client.ASRWebsocketClient().send_audio("ws://localhost:8765", b"dummy")
        assert transcript == "hello world"
    finally:
        server.close()
        await server.wait_closed()
