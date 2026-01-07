#!/bin/bash

# OM1 啟動腳本

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "啟動 OM1 Spot Agent..."
echo "WebSim 介面: http://localhost:8000"
echo "按 Ctrl+C 停止"
echo ""

$HOME/.local/bin/uv run src/run.py spot
