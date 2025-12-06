from flask import Flask, request, jsonify, Response
from om1_mujoco_adapter import MujocoAdapter
import time
import atexit

# --- Configuration ---
HOST = '127.0.0.1'
PORT = 8080

# 1. Initialize the MuJoCo adapter globally
MUJOCO_ADAPTER = None
try:
    MUJOCO_ADAPTER = MujocoAdapter()
except Exception as e:
    print(f"FATAL ERROR: Could not initialize MujocoAdapter: {e}")
    exit(1)

# Cleanup function to close simulation threads on exit
@atexit.register
def shutdown_adapter():
    if MUJOCO_ADAPTER:
        MUJOCO_ADAPTER.close()

# 2. Initialize the Flask App
app = Flask(__name__)

# --- API Endpoint for OM1 Actions ---
@app.route('/control', methods=['POST'])
def mujoco_control_action():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    command_data = request.get_json()
    result = MUJOCO_ADAPTER.handle_control_command(command_data)
    return jsonify({"status": result['status'], "result": result['result']}), 200

# --- MJPEG Streaming Logic (Now PNG) ---
def generate_frames(adapter):
    while adapter.is_running:
        frame = adapter.get_latest_frame()
        # Note: We still use multipart/x-mixed-replace, but stream PNG bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
        # Stream at 5 FPS to keep the plot smooth and low-resource
        time.sleep(1/5) 

@app.route('/stream')
def video_stream():
    """This endpoint streams the Matplotlib plot as a live feed."""
    return Response(generate_frames(MUJOCO_ADAPTER),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Main Entry point ---
if __name__ == '__main__':
    print(f"Starting MuJoCo Control API Server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
