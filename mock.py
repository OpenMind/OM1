"""Mock face presence service for local testing on Mac.

Run: python scripts/mock_face_service.py
Press Enter to simulate a face appearing/disappearing.

Starts with no face. Each Enter toggles face on/off.
"""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

face_present = False


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/who":
            if face_present:
                body = {
                    "faces": [
                        {"name": "unknown", "bbox": [200, 100, 400, 350], "area": 50000, "track_id": 1}
                    ],
                    "server_ts": time.time(),
                    "frame_hw": [480, 640],
                }
            else:
                body = {"faces": [], "server_ts": time.time(), "frame_hw": [480, 640]}

            resp = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        print(format % args)


def main():
    global face_present
    server = HTTPServer(("127.0.0.1", 6793), Handler)
    server.timeout = 0.1

    print("Mock face service running on http://127.0.0.1:6793")
    print("Face: OFF")
    print("Press Enter to toggle face on/off, Ctrl+C to quit\n")

    import threading
    def serve():
        while True:
            server.handle_request()

    t = threading.Thread(target=serve, daemon=True)
    t.start()

    try:
        while True:
            input()
            face_present = not face_present
            status = "ON  (area=50000)" if face_present else "OFF"
            print(f"Face: {status}")
    except (KeyboardInterrupt, EOFError):
        print("\nStopping.")


if __name__ == "__main__":
    main()

