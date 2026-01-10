"""
JSON5 linting script.

This script validates all `.json5` files in the repository to ensure they
are syntactically correct. It is intended to be used both locally and in CI.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import json5


def validate_json5_file(file_path: Path) -> Tuple[bool, str]:
    """
    Validate a single JSON5 file.

    Parameters
    ----------
    file_path : Path
        Path to the JSON5 file to validate.

    Returns
    -------
    Tuple[bool, str]
        A tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            json5.load(f)
        return True, ""
    except json5.JSON5DecodeError as e:
        # Extract line number if available
        error_msg = str(e)
        if hasattr(e, "lineno") and e.lineno:
            error_msg = f"Line {e.lineno}: {error_msg}"
        return False, error_msg
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def main() -> int:
    """
    Validate all JSON5 files in the repository.

    Returns
    -------
    int
        Exit code: 0 if all files are valid, 1 if any file is invalid.
    """
    root = Path(".")
    json5_files = sorted(root.rglob("*.json5"))

    if not json5_files:
        print("No JSON5 files found.")
        return 0

    print(f"Validating {len(json5_files)} JSON5 file(s)...")

    failed_files: List[Tuple[Path, str]] = []

    for file in json5_files:
        is_valid, error_msg = validate_json5_file(file)
        if not is_valid:
            failed_files.append((file, error_msg))
            print(f"[ERROR] Invalid JSON5: {file}")
            print(f"        {error_msg}")

    if failed_files:
        print(f"\nJSON5 linting failed: {len(failed_files)} file(s) have errors.")
        return 1

    print(f"✓ All {len(json5_files)} JSON5 file(s) are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
