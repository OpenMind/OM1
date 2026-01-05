import sys
from pathlib import Path
import json5

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
FAILED = False

if not CONFIG_DIR.exists():
    print(f"Config directory not found: {CONFIG_DIR}")
    sys.exit(1)

json5_files = list(CONFIG_DIR.rglob("*.json5"))

if not json5_files:
    print("No JSON5 files found.")
    sys.exit(0)

for file in json5_files:
    try:
        with file.open("r", encoding="utf-8") as f:
            json5.load(f)
        print(f"✔ Valid JSON5: {file}")
    except Exception as e:
        FAILED = True
        print(f"✖ Invalid JSON5: {file}")
        print(f"  Error: {e}")

if FAILED:
    sys.exit(1)

print("All JSON5 config files are valid.")
