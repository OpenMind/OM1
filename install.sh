#!/bin/bash

# OM1 一鍵安裝腳本
# OpenMind AI Robot Runtime

set -e

echo "======================================"
echo "  OM1 安裝腳本 - OpenMind AI Runtime"
echo "======================================"
echo ""

# 檢測作業系統
OS="$(uname -s)"
case "$OS" in
    Darwin*) OS_TYPE="mac" ;;
    Linux*)  OS_TYPE="linux" ;;
    *)       echo "不支援的作業系統: $OS"; exit 1 ;;
esac

echo "[1/5] 安裝 uv 套件管理器..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "✓ uv 已安裝"
else
    echo "✓ uv 已存在"
fi

echo ""
echo "[2/5] 安裝系統依賴項..."
if [ "$OS_TYPE" = "mac" ]; then
    if command -v brew &> /dev/null; then
        brew install portaudio ffmpeg 2>/dev/null || echo "✓ 依賴項已存在"
    else
        echo "請先安裝 Homebrew: https://brew.sh"
        exit 1
    fi
elif [ "$OS_TYPE" = "linux" ]; then
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev python3-dev ffmpeg
fi
echo "✓ 系統依賴項已安裝"

echo ""
echo "[3/5] 初始化 Git 子模組..."
git submodule update --init
echo "✓ 子模組已初始化"

echo ""
echo "[4/5] 建立 Python 虛擬環境..."
$HOME/.local/bin/uv venv
echo "✓ 虛擬環境已建立"

echo ""
echo "[5/5] 設定 API 金鑰..."
if [ ! -f .env ]; then
    cp env.example .env
fi

# 詢問 API 金鑰
echo ""
echo "請輸入你的 OM1 API 金鑰（從 https://portal.openmind.org 取得）："
read -r API_KEY

if [ -n "$API_KEY" ]; then
    if [ "$OS_TYPE" = "mac" ]; then
        sed -i '' "s/^OM_API_KEY=.*/OM_API_KEY=$API_KEY/" .env
    else
        sed -i "s/^OM_API_KEY=.*/OM_API_KEY=$API_KEY/" .env
    fi
    echo "✓ API 金鑰已設定"
else
    echo "⚠ 未輸入 API 金鑰，請稍後手動編輯 .env 檔案"
fi

echo ""
echo "======================================"
echo "  安裝完成！"
echo "======================================"
echo ""
echo "執行方式："
echo "  $HOME/.local/bin/uv run src/run.py spot"
echo ""
echo "然後開啟瀏覽器訪問："
echo "  http://localhost:8000"
echo ""
