import asyncio
import json

import websockets

URI = "ws://127.0.0.1:8765"


async def run():
    async with websockets.connect(URI) as ws:
        await ws.send(json.dumps({"op": "reset"}))
        print(await ws.recv())

        await ws.send(json.dumps({"op": "set", "target": {"j1": 20, "j2": -45}}))
        print(await ws.recv())

        await ws.send(json.dumps({"op": "step", "n": 200}))
        print(await ws.recv())

        await ws.send(json.dumps({"op": "state"}))
        print(await ws.recv())


asyncio.run(run())
