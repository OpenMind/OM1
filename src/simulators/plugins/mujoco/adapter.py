import asyncio
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import websockets


class HTTPAdapter:
    """HTTP server adapter for MuJoCo control"""

    def __init__(self, ws_url="ws://127.0.0.1:8765", http_port=8088):
        self.ws_url = ws_url
        self.http_port = http_port
        self.server = None

    class Handler(BaseHTTPRequestHandler):
        """HTTP request handler"""

        ws_url: str | None = None  # Will be set by outer class

        def _send(self, code, payload):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_POST(self):
            if self.path != "/control":
                return self._send(404, {"ok": False, "error": "not found"})
            try:
                ln = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(ln)
                data = json.loads(body or "{}")
                j1 = float(data.get("j1", 0))
                j2 = float(data.get("j2", 0))
                steps = int(data.get("steps", 200))

                if self.ws_url is None:
                    raise ValueError("WebSocket URL not set")
                ws_url: str = self.ws_url

                async def go():
                    async with websockets.connect(ws_url) as ws:
                        await ws.send(json.dumps({"op": "reset"}))
                        await ws.recv()
                        await ws.send(
                            json.dumps({"op": "set", "target": {"j1": j1, "j2": j2}})
                        )
                        await ws.recv()
                        await ws.send(json.dumps({"op": "step", "n": steps}))
                        return json.loads(await ws.recv())

                obs = asyncio.run(go())
                return self._send(200, {"ok": True, "result": obs})
            except Exception as e:
                return self._send(400, {"ok": False, "error": str(e)})

        def log_message(self, format, *args):
            """Suppress default logging"""
            pass

    def start_server(self):
        """Start the HTTP server"""
        # Set the ws_url on the Handler class
        self.Handler.ws_url = self.ws_url

        self.server = HTTPServer(("0.0.0.0", self.http_port), self.Handler)
        print(
            f"[HTTP Adapter] Listening on http://0.0.0.0:{self.http_port} -> WebSocket {self.ws_url}"
        )
        self.server.serve_forever()

    def run(self):
        """Run the HTTP server in current thread"""
        self.start_server()

    def stop(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
