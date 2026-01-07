#!/bin/bash
set -e

echo "======================================"
echo "       🚀 OM1 自动安装脚本 v3.2"
echo "======================================"

OM1_PATH="$HOME/OM1"

# -------------------------------
# 1) 安装 uv
# -------------------------------
echo "[1/8] 检查 uv 是否存在..."
if ! command -v uv &>/dev/null; then
    echo "未检测到 uv，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
    export PATH=$HOME/.local/bin:$PATH

    if [ -f "$HOME/.local/bin/uv" ]; then
        ln -sf $HOME/.local/bin/uv /usr/local/bin/uv
        ln -sf $HOME/.local/bin/uvx /usr/local/bin/uvx
    fi

    echo "uv 安装成功（PATH 已修复）"
else
    echo "uv 已存在，跳过安装"
fi

if ! command -v uv &>/dev/null; then
    echo "❌ uv 仍不可用，终止安装"
    exit 1
fi

# -------------------------------
# 2) 克隆仓库
# -------------------------------
echo
echo "[2/8] 克隆 OM1 仓库..."
if [ -d "$OM1_PATH" ]; then
    echo "检测到 ~/OM1 已存在，跳过克隆"
else
    git clone https://github.com/OpenManus/OM1.git "$OM1_PATH"
fi

cd "$OM1_PATH"

# -------------------------------
echo "[3/8] 初始化子模块..."
git submodule update --init --recursive

# -------------------------------
echo
echo "[4/8] 创建 uv 虚拟环境..."
rm -rf .venv || true
uv venv

# -------------------------------
echo
echo "[5/8] 激活虚拟环境..."
source .venv/bin/activate

# -------------------------------
echo
echo "[6/8] 创建 .env..."

rm -f .env
touch .env

read -p "请输入 ETH 地址: " ETH_ADDRESS
echo "ETH_ADDRESS=\"$ETH_ADDRESS\"" >> .env

read -p "请输入 OM_API_KEY: " OM_API_KEY
echo "OM_API_KEY=\"$OM_API_KEY\"" >> .env

read -p "请输入机器人 ID (URID): " URID
echo "URID=\"$URID\"" >> .env

echo "生成的 .env 内容如下："
cat .env

# -------------------------------
echo
echo "[7/8]（跳过）无需安装 requirements.txt，OM1 无依赖文件"

# -------------------------------
echo
echo "[8/8] 完成！"

echo "======================================"
echo " OM1 安装完成！你现在可以执行："
echo
echo "  cd ~/OM1"
echo "  source .venv/bin/activate"
echo "  uv run src/run.py conversation"
echo
echo "======================================"
