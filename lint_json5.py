#!/usr/bin/env python3
import sys
import os
import json5
from pathlib import Path

def lint_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        data = json5.loads(content)
        print(f"✅ {filepath}")
        return True
    except json5.JSON5DecodeError as e:
        print(f"❌ {filepath}:{e.lineno}:{e.colno}")
        print(f"   Error: {e.msg}")
        print(f"   Near: {e.value!r}")
        return False
    except Exception as e:
        print(f"❌ {filepath}: {str(e)}")
        return False

def main():
    root = Path.cwd()
    json5_files = list(root.glob("**/*.json5"))
    
    if not json5_files:
        print("No JSON5 files found.")
        sys.exit(0)
    
    print(f"Found {len(json5_files)} JSON5 files to lint")
    
    failed = 0
    for file_path in json5_files:
        if not lint_file(file_path):
            failed += 1
    
    if failed > 0:
        print(f"\n❌ {failed} JSON5 files failed linting")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(json5_files)} JSON5 files are valid!")
        sys.exit(0)

if __name__ == "__main__":
    main()
