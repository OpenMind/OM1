import os
import sys

# Force Python to load zenoh_msgs from src/ instead of .venv
src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
src_path = os.path.abspath(src_path)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Remove cached zenoh_msgs from sys.modules to force reimport from src/
for key in list(sys.modules.keys()):
    if "zenoh_msgs" in key:
        del sys.modules[key]
