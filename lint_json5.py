#!/usr/bin/env python3
"""JSON5 linter for OpenMind configuration files."""
import json5
import sys
from pathlib import Path
def lint_file(filepath: str) -> bool:
    """Validate JSON5 file syntax and structure."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            json5.loads(f.read())
        print(f"✅ {filepath}")
        return True
    except json5.JSON5DecoderException as e:
        print(f"❌ {filepath}: {e}")
        return False
    except Exception as e:
        print(f"⚠️ {filepath}: {e}")
        return False
def main() -> None:
    """Lint all JSON5 files in current directory and subdirectories."""
    root = Path.cwd()
    json5_files = list(root.glob("**/*.json5"))
    if not json5_files:
        print("No JSON5 files found.")
        return
    errors = sum(1 for f in json5_files if not lint_file(str(f)))
    sys.exit(1 if errors else 0)
if __name__ == "__main__":
    main()
