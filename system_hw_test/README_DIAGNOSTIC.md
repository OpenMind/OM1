# OM1 Diagnostic Tool

A simple diagnostic script to check if your system is ready to run OM1.

## Usage

Run the diagnostic tool:
```bash
python om1_doctor.py
```

## What it checks

- ✓ Python version (3.8+)
- ✓ Required packages from requirements.txt
- ✓ Configuration files in config/
- ✓ GPU availability (optional)

## Example Output
```
==================================================
  OM1 Diagnostic Tool
==================================================

✓ Python Version: PASS
  → Python 3.14.2

✗ requirements.txt: FAIL
  → File not found

✓ NVIDIA GPU: PASS
  → GPU detected
```

## Requirements

- Python 3.8 or higher
- Works on Windows, Linux, and macOS