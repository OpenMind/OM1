#!/usr/bin/env bash

echo "===================================="
echo "     OM1 Environment Diagnostics"
echo "===================================="

# Check OS
echo -e "\n[1] Checking Operating System..."
uname -a

# Python version
echo -e "\n[2] Checking Python version..."
PY_VER=$(python3 --version 2>/dev/null)
if [[ "$PY_VER" == *"3.10"* ]]; then
  echo "✔ Python version OK: $PY_VER"
else
  echo "✘ Python version is NOT 3.10 (recommended)"
  echo "  Detected: $PY_VER"
fi

# Check uv
echo -e "\n[3] Checking uv package manager..."
if command -v uv >/dev/null 2>&1; then
  echo "✔ uv is installed"
else
  echo "✘ uv is NOT installed"
  echo "  Install: pip install uv"
fi

# Check ffmpeg
echo -e "\n[4] Checking FFmpeg..."
if command -v ffmpeg >/dev/null 2>&1; then
  echo "✔ ffmpeg installed"
else
  echo "✘ ffmpeg missing"
fi

# Check PortAudio
echo -e "\n[5] Checking PortAudio..."
if pkg-config --exists portaudio-2.0 2>/dev/null; then
  echo "✔ PortAudio detected"
else
  echo "✘ PortAudio missing"
fi

# Docker
echo -e "\n[6] Checking Docker..."
if command -v docker >/dev/null 2>&1; then
  echo "✔ Docker installed"
else
  echo "✘ Docker not installed"
fi

# Docker running
echo -e "\n[7] Checking Docker daemon..."
docker info >/dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "✔ Docker daemon is running"
else
  echo "✘ Docker daemon is NOT running"
fi

# Docker permissions
echo -e "\n[8] Checking Docker permissions..."
if docker ps >/dev/null 2>&1; then
  echo "✔ Docker permissions OK"
else
  echo "✘ User does NOT have permission to run Docker"
fi

# Git submodules
echo -e "\n[9] Checking git submodules..."
if [ -d "unitree_sdk" ] || [ -d "src" ]; then
  echo "✔ Submodules appear initialized"
else
  echo "✘ Submodules missing"
  echo "  Run: git submodule update --init --recursive"
fi

echo -e "\n===================================="
echo "         Diagnostics complete"
echo "===================================="
