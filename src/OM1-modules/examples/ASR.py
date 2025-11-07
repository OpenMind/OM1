import logging
from om1_speech import AudioInputStream
from om1_utils import ws
from src.actions.home_assistant import perform_action  # ✅ Correct import path

# --- Setup logger ---
root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)
logging.basicConfig(level=logging.INFO)

# --- Handle ASR messages from OM1 ---
def on_message(message):
    """Handle recognized speech results coming from OM1 ASR."""
    try:
        if isinstance(message, dict) and "text" in message:
            recognized_text = message["text"]
            print(f"[ASR] You said: {recognized_text}")
            perform_action(recognized_text)  # ✅ Send command to Home Assistant
        else:
            print("[ASR] No recognized text in message:", message)
    except Exception as e:
        print("[ASR] Error processing message:", e)

# --- Connect to OM1 ASR WebSocket ---
ws_client = ws.Client(
    url="wss://api-asr.openmind.org",
    on_message=on_message
)
ws_client.start()

# --- Capture audio from microphone and stream to OM1 ASR ---
audio_stream_input = AudioInputStream(audio_data_callback=ws_client.send_message)
audio_stream_input.start()

# --- Keep the program running ---
while True:
    pass
