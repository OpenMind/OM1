import shutil
import sys

def check_command(cmd: str):
    return shutil.which(cmd) is not None

def ensure_environment(mode: str):
    missing = []

    if not check_command("uv"):
        missing.append("uv")

    if mode == "spot":
        if not check_command("ffmpeg"):
            missing.append("ffmpeg")

    if missing:
        print("\n[Environment Check Failed]")
        print("Missing dependencies:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease install them before running this mode.")
sys.exit(1)
