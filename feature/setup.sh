#!/bin/bash

# OM1 Setup Script - Automated Installation
# This script automates the OM1 installation process across macOS, Linux, and Windows
# Usage: bash setup.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.10"
UV_VERSION="latest"
PROJECT_NAME="OM1"
ENV_FILE=".env"
CONFIG_DIR="config"

# Functions
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
    echo "$OS"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python() {
    print_header "Checking Python Installation"
    
    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        print_info "Please install Python 3.10+ from https://www.python.org/downloads/"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
    
    # Check if version is >= 3.10
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
        print_error "Python 3.10 or higher required (current: $PYTHON_VERSION)"
        exit 1
    fi
}

# Install uv package manager
install_uv() {
    print_header "Installing/Checking UV Package Manager"
    
    if command_exists uv; then
        UV_INSTALLED=$(uv --version)
        print_success "UV already installed: $UV_INSTALLED"
    else
        print_info "Installing UV..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
        # Add uv to PATH for current session
        if [[ "$OS" == "linux" ]] || [[ "$OS" == "macos" ]]; then
            export PATH="$HOME/.local/bin:$PATH"
        fi
        
        if command_exists uv; then
            print_success "UV installed successfully"
        else
            print_error "Failed to install UV"
            exit 1
        fi
    fi
}

# Install system dependencies based on OS
install_system_deps() {
    print_header "Installing System Dependencies"
    
    case "$OS" in
        macos)
            print_info "Detected macOS - Installing dependencies via Homebrew"
            
            if ! command_exists brew; then
                print_warning "Homebrew not found. Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            print_info "Installing portaudio and ffmpeg..."
            brew install portaudio ffmpeg
            print_success "macOS dependencies installed"
            ;;
            
        linux)
            print_info "Detected Linux - Installing dependencies via apt"
            
            if ! command_exists apt-get; then
                print_error "apt-get not found. Please install dependencies manually:"
                print_info "sudo apt-get update"
                print_info "sudo apt-get install portaudio19-dev python3-dev ffmpeg"
                exit 1
            fi
            
            print_info "Updating package lists..."
            sudo apt-get update
            
            print_info "Installing portaudio, python3-dev, and ffmpeg..."
            sudo apt-get install -y portaudio19-dev python3-dev ffmpeg
            print_success "Linux dependencies installed"
            ;;
            
        windows)
            print_warning "Windows detected"
            print_info "Please install the following manually using your package manager (chocolatey, winget, or manually):"
            print_info "  - portaudio: choco install portaudio"
            print_info "  - ffmpeg: choco install ffmpeg"
            print_info "Or download from: https://ffmpeg.org/download.html"
            read -p "Press enter once you've installed these dependencies..."
            ;;
            
        *)
            print_error "Unknown OS: $OS"
            print_warning "Please install portaudio and ffmpeg manually"
            read -p "Press enter once you've installed these dependencies..."
            ;;
    esac
}

# Clone repository and submodules
setup_repo() {
    print_header "Setting Up Repository"
    
    if [ -d ".git" ]; then
        print_info "Repository already initialized"
    else
        print_error "Not in OM1 repository directory"
        print_info "Please run this script from the OM1 root directory"
        exit 1
    fi
    
    print_info "Initializing and updating git submodules..."
    git submodule update --init --recursive
    print_success "Repository configured"
}

# Create virtual environment
setup_venv() {
    print_header "Setting Up Virtual Environment"
    
    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists at .venv"
        read -p "Do you want to recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf .venv
            uv venv
        fi
    else
        print_info "Creating virtual environment..."
        uv venv
    fi
    
    print_success "Virtual environment created"
}

# Install Python dependencies
install_dependencies() {
    print_header "Installing Python Dependencies"
    
    print_info "This may take a few minutes..."
    uv pip install -e .
    
    print_success "Python dependencies installed"
}

# Setup environment variables
setup_env() {
    print_header "Setting Up Environment Variables"
    
    if [ -f "$ENV_FILE" ]; then
        print_warning ".env file already exists"
        print_info "Current content:"
        cat "$ENV_FILE"
        read -p "Do you want to overwrite it? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping .env setup"
            return
        fi
    fi
    
    print_info "Creating .env file..."
    cat > "$ENV_FILE" << 'EOF'
# OM1 Environment Configuration
# Fill in your API keys and configuration here

# OpenMind API Key (required)
# Get your key from: https://openmind.org
OM_API_KEY=your_api_key_here

# OpenAI Configuration (optional, defaults to OpenMind endpoint)
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o

# Other LLM Providers (optional)
# DEEPSEEK_API_KEY=your_deepseek_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here

# Robot Hardware Configuration (optional)
# ROBOT_TYPE=spot
# ROBOT_HOST=192.168.1.100
# ROBOT_PORT=5005

# Debug and Logging (optional)
DEBUG=false
LOG_LEVEL=INFO
EOF
    
    print_success ".env file created"
    print_info "Please add your API keys to: $ENV_FILE"
}

# Validate configuration
validate_setup() {
    print_header "Validating Setup"
    
    local valid=true
    
    # Check Python
    if command_exists python3; then
        print_success "Python 3 available"
    else
        print_error "Python 3 not found"
        valid=false
    fi
    
    # Check UV
    if command_exists uv; then
        print_success "UV package manager available"
    else
        print_error "UV not found"
        valid=false
    fi
    
    # Check virtual environment
    if [ -d ".venv" ]; then
        print_success "Virtual environment created"
    else
        print_error "Virtual environment not found"
        valid=false
    fi
    
    # Check .env
    if [ -f "$ENV_FILE" ]; then
        if grep -q "your_api_key_here" "$ENV_FILE"; then
            print_warning ".env exists but API_KEY not configured"
        else
            print_success ".env configured"
        fi
    else
        print_warning ".env file not found"
    fi
    
    return 0
}

# Print next steps
print_next_steps() {
    print_header "Setup Complete! 🎉"
    
    echo -e "${GREEN}OM1 is ready to use!${NC}\n"
    
    echo "Next steps:"
    echo -e "  1. ${YELLOW}Add your API key to .env:${NC}"
    echo "     OM_API_KEY=your_key_from_openmind.org"
    echo ""
    echo -e "  2. ${YELLOW}Activate virtual environment:${NC}"
    if [[ "$OS" == "windows" ]]; then
        echo "     .venv\\Scripts\\activate"
    else
        echo "     source .venv/bin/activate"
    fi
    echo ""
    echo -e "  3. ${YELLOW}Run the Spot agent:${NC}"
    echo "     uv run src/run.py spot"
    echo ""
    echo -e "  4. ${YELLOW}View WebSim dashboard:${NC}"
    echo "     http://localhost:8000/"
    echo ""
    echo -e "${BLUE}For detailed documentation:${NC}"
    echo "  - https://docs.openmind.org"
    echo "  - GitHub: https://github.com/OpenMind/OM1"
    echo ""
}

# Main execution
main() {
    clear
    
    echo -e "${BLUE}"
    echo " ██████╗ ███╗   ███╗ ██╗"
    echo "██╔═══██╗████╗ ████║ ██║"
    echo "██║   ██║██╔████╔██║ ██║"
    echo "██║   ██║██║╚██╔╝██║ ╚═╝"
    echo "╚██████╔╝██║ ╚═╝ ██║ ██╗"
    echo " ╚═════╝ ╚═╝     ╚═╝ ╚═╝"
    echo -e "${NC}"
    
    print_header "OM1 Automated Setup"
    print_info "This script will install OM1 and all dependencies"
    
    # Detect OS
    OS=$(detect_os)
    print_info "Detected OS: $OS"
    
    if [ "$OS" = "unknown" ]; then
        print_error "Unknown operating system"
        exit 1
    fi
    
    # Run setup steps
    check_python
    install_uv
    install_system_deps
    setup_repo
    setup_venv
    install_dependencies
    setup_env
    validate_setup
    print_next_steps
}

# Run main function
main "$@"
