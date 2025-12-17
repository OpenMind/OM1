import asyncio
import json
import os

import websockets

from .mj_env import MJEnv

HOST = os.getenv("OM1_BRIDGE_HOST", "0.0.0.0")
PORT = int(os.getenv("OM1_BRIDGE_PORT", "8765"))
MODEL = os.getenv("OM1_MODEL", "assets/simple_arm.xml")


class MuJoCoStepper:
    """WebSocket server for MuJoCo simulation stepping"""

    def __init__(self, model_path, host="0.0.0.0", port=8765):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.env = MJEnv(model_path)
        self.env.reset()

    async def handle(self, ws):
        async for msg in ws:
            try:
                req = json.loads(msg)
                op = req.get("op")

                if op == "reset":
                    obs = self.env.reset()
                    await ws.send(json.dumps({"ok": True, "obs": obs}))
                elif op == "set":
                    target = req.get("target", {})
                    self.env.set_target_qpos(target)
                    await ws.send(json.dumps({"ok": True}))
                elif op == "step":
                    n = int(req.get("n", 1))
                    obs = self.env.step(n)
                    await ws.send(json.dumps({"ok": True, "obs": obs}))
                elif op == "state":
                    await ws.send(json.dumps({"ok": True, "obs": self.env._obs()}))
                else:
                    await ws.send(json.dumps({"ok": False, "error": "unknown op"}))
            except Exception as e:
                await ws.send(json.dumps({"ok": False, "error": str(e)}))

    async def start_server(self):
        async with websockets.serve(self.handle, self.host, self.port):
            print(
                f"[MuJoCo Bridge] WebSocket server running at ws://{self.host}:{self.port}"
            )
            print(f"[MuJoCo Bridge] Model: {self.model_path}")
            await asyncio.Future()  # run forever

    def run(self):
        """Run the WebSocket server"""
        asyncio.run(self.start_server())
