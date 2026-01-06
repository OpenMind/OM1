import sys
from pathlib import Path

import json5


def main():
    root = Path(".")
    json5_files = list(root.rglob("*.json5"))

    if not json5_files:
        print("No JSON5 files found.")
        return 0

    failed = False

    for file in json5_files:
        try:
            with file.open("r", encoding="utf-8") as f:
                json5.load(f)
        except Exception as e:
            failed = True
            print(f"[ERROR] Invalid JSON5: {file}")
            print(f"        {e}")

    if failed:
        print("\nJSON5 linting failed.")
        return 1

    print("All JSON5 files are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
