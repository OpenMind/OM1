# app_presence_demo.py
import time

from inputs.plugins.face_presence_input import FacePresenceInput
from providers.face_presence_provider import FacePresenceProvider

BASE_URL = "http://127.0.0.1:6793"  # your run.py HTTP host/port

# Start the singleton provider (polls /who at ~5Hz by default)
provider = FacePresenceProvider.instance(
    base_url=BASE_URL,  # server base, e.g. http://127.0.0.1:6793
    recent_sec=2.0,  # lookback window for presence
    fps=5.0,  # polling rate (Hz)
    timeout_s=1.5,
)
provider.start()

# Create an input facade
inp = FacePresenceInput(provider)

# Give it a tick to fetch at least once (optional)
time.sleep(0.4)

# When you need to build a prompt, PULL the latest presence:
reading = inp.peek()  # doesn't clear the buffer
# reading = inp.get_latest()  # clears backlog, returns newest only

if reading:
    # reading.text is already formatted: e.g. "present: [Alice], unknown=1 @ 1696352905.123"
    print("Presence for LLM:", reading.text)
else:
    print("Presence for LLM: no data yet")

last10 = provider.get_history(10)
print(last10)


"""react to changes
try:
    while True:
        item = provider.buffer.wait_next(last_seq=last_seq, timeout=5.0)
        if item is None:
            # no change in 5s — optional: show current (may be stale)
            current = provider.buffer.peek_latest()
            if current:
                print("(idle) current:", current.value.to_text())
            continue

        # got a newer snapshot
        print("CHANGED:", item.value.to_text())
        last_seq = item.seq
finally:
    provider.stop
"""
# later, when shutting down:
# provider.stop()
