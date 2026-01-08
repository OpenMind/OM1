#!/bin/bash
# OM1 Quickstart Script
# Sets up environment and starts OM1 with common configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}OpenMind OM1 Quickstart${NC}"
echo "=========================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

# Check uv
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv not found. Installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env << 'ENVEOF'
# OpenMind API Configuration
OM_API_KEY=your_api_key_here
URID=your_urid_here

# Optional: Ethereum wallet
# ETH_ADDRESS=your_wallet_address
ENVEOF
    echo -e "${YELLOW}Please edit .env with your API credentials${NC}"
fi

# Validate config
echo -e "\n${GREEN}Validating configuration...${NC}"
if [ -f "scripts/validate_config.py" ]; then
    uv run python scripts/validate_config.py config/spot.json5
else
    echo -e "${YELLOW}Config validation script not found, skipping...${NC}"
fi

# Estimate costs
echo -e "\n${GREEN}Estimating API costs...${NC}"
if [ -f "scripts/api_cost_estimator.py" ]; then
    uv run python scripts/api_cost_estimator.py config/spot.json5
else
    echo -e "${YELLOW}Cost estimator script not found, skipping...${NC}"
fi

# Ask to start
echo -e "\n${GREEN}Setup complete!${NC}"
echo -n "Start OM1 now? (y/n) "
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${GREEN}Starting OM1...${NC}"
    export PATH="$HOME/.local/bin:$PATH"
    source .env
    uv run src/run.py spot
else
    echo -e "\nTo start OM1 later, run:"
    echo -e "${YELLOW}  source .env${NC}"
    echo -e "${YELLOW}  uv run src/run.py spot${NC}"
fi
