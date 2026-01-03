#!/usr/bin/env python3
"""
OM1 Setup Script - Automated Installation
Cross-platform setup tool for OM1 installation
Usage: python3 setup.py
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Optional
import json

# Colors for output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}=== {text} ==={Colors.END}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def detect_os() -> str:
    """Detect operating system"""
    system = platform.system()
    if system == "Darwin":
        return "macos"
    elif system == "Linux":
        return "linux"
    elif system == "Windows":
        return "windows"
    else:
        return "unknown"

def command_exists(cmd: str) -> bool:
    """Check if command exists in PATH"""
    return shutil.which(cmd) is not None

def run_command(cmd: list, description: str = "", check: bool = True) -> Tuple[int, str]:
    """
    Run a shell command
    
    Args:
        cmd: Command as list
        description: Description of what command does
        check: Whether to raise exception on error
    
    Returns:
        Tuple of (return_code, output)
    """
    if description:
        print_info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode, result.stdout + result.stderr
    except Exception as e:
        if check:
            print_error(f"Failed to run command: {e}")
            sys.exit(1)
        return 1, str(e)

def check_python() -> None:
    """Check Python version"""
    print_header("Checking Python Installation")
    
    if not command_exists("python3"):
        print_error("Python 3 is not installed")
        print_info("Please install Python 3.10+ from https://www.python.org/downloads/")
        sys.exit(1)
    
    # Get Python version
    code, output = run_command(
        ["python3", "--version"],
        check=False
    )
    
    print_success(f"Python {output.strip()} found")
    
    # Check version >= 3.10
    version_check = run_command(
        ["python3", "-c", "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"],
        check=False
    )
    
    if version_check[0] != 0:
        print_error("Python 3.10 or higher required")
        sys.exit(1)

def install_uv() -> None:
    """Install UV package manager if not present"""
    print_header("Installing/Checking UV Package Manager")
    
    if command_exists("uv"):
        code, output = run_command(["uv", "--version"], check=False)
        print_success(f"UV already installed: {output.strip()}")
        return
    
    print_info("Installing UV...")
    
    if detect_os() == "windows":
        # Windows installation
        code, output = run_command(
            ["python3", "-m", "pip", "install", "uv"],
            check=False
        )
    else:
        # Unix installation
        code, output = run_command(
            ["curl", "-LsSf", "https://astral.sh/uv/install.sh"],
            check=False
        )
        if code == 0:
            run_command(["sh"], check=False)
    
    # Verify installation
    if command_exists("uv"):
        print_success("UV installed successfully")
    else:
        print_error("Failed to install UV")
        print_info("Try installing manually: pip install uv")
        sys.exit(1)

def install_system_deps(os_type: str) -> None:
    """Install system dependencies based on OS"""
    print_header("Installing System Dependencies")
    
    if os_type == "macos":
        print_info("Detected macOS")
        
        if not command_exists("brew"):
            print_warning("Homebrew not found. Installing...")
            code, _ = run_command(
                ["/bin/bash", "-c", 
                 "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"],
                check=False
            )
            if code != 0:
                print_error("Failed to install Homebrew")
                sys.exit(1)
        
        print_info("Installing portaudio and ffmpeg...")
        run_command(["brew", "install", "portaudio", "ffmpeg"])
        print_success("macOS dependencies installed")
        
    elif os_type == "linux":
        print_info("Detected Linux")
        
        if not command_exists("apt-get"):
            print_error("apt-get not found")
            print_info("Please install dependencies manually:")
            print_info("  sudo apt-get update")
            print_info("  sudo apt-get install portaudio19-dev python3-dev ffmpeg")
            sys.exit(1)
        
        print_info("Updating package lists...")
        run_command(["sudo", "apt-get", "update"])
        
        print_info("Installing dependencies...")
        run_command([
            "sudo", "apt-get", "install", "-y",
            "portaudio19-dev", "python3-dev", "ffmpeg"
        ])
        print_success("Linux dependencies installed")
        
    elif os_type == "windows":
        print_warning("Windows detected")
        print_info("Please install the following dependencies:")
        print_info("  Option 1 (Chocolatey):")
        print_info("    choco install portaudio ffmpeg")
        print_info("  Option 2 (WinGet):")
        print_info("    winget install -e --id FFmpeg.FFmpeg")
        print_info("  Option 3 (Manual):")
        print_info("    https://ffmpeg.org/download.html")
        input("\nPress Enter once you've installed these dependencies...")

def setup_repo() -> None:
    """Setup git repository and submodules"""
    print_header("Setting Up Repository")
    
    if not Path(".git").exists():
        print_error("Not in OM1 repository directory")
        print_info("Please run this script from the OM1 root directory")
        sys.exit(1)
    
    print_info("Initializing and updating git submodules...")
    run_command(["git", "submodule", "update", "--init", "--recursive"])
    print_success("Repository configured")

def setup_venv() -> None:
    """Create virtual environment"""
    print_header("Setting Up Virtual Environment")
    
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print_warning("Virtual environment already exists")
        response = input("Do you want to recreate it? (y/n): ").lower()
        if response == 'y':
            shutil.rmtree(venv_path)
            run_command(["uv", "venv"])
        else:
            print_info("Skipping venv creation")
            return
    else:
        print_info("Creating virtual environment...")
        run_command(["uv", "venv"])
    
    print_success("Virtual environment created")

def install_dependencies() -> None:
    """Install Python dependencies"""
    print_header("Installing Python Dependencies")
    
    print_info("This may take a few minutes...")
    run_command(["uv", "pip", "install", "-e", "."])
    print_success("Python dependencies installed")

def setup_env() -> None:
    """Setup environment variables"""
    print_header("Setting Up Environment Variables")
    
    env_file = Path(".env")
    
    if env_file.exists():
        print_warning(".env file already exists")
        print_info(f"Current content:\n{env_file.read_text()}")
        response = input("Do you want to overwrite it? (y/n): ").lower()
        if response != 'y':
            print_info("Skipping .env setup")
            return
    
    env_content = """# OM1 Environment Configuration
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
"""
    
    env_file.write_text(env_content)
    print_success(".env file created")
    print_info("Please add your API keys to: .env")

def validate_setup() -> None:
    """Validate the setup"""
    print_header("Validating Setup")
    
    checks = [
        ("Python 3", lambda: command_exists("python3")),
        ("UV Package Manager", lambda: command_exists("uv")),
        ("Virtual Environment", lambda: Path(".venv").exists()),
        (".env Configuration", lambda: Path(".env").exists()),
    ]
    
    for check_name, check_fn in checks:
        if check_fn():
            print_success(f"{check_name} available")
        else:
            print_warning(f"{check_name} not found")

def print_next_steps(os_type: str) -> None:
    """Print next steps for user"""
    print_header("Setup Complete! 🎉")
    
    print(f"{Colors.GREEN}OM1 is ready to use!{Colors.END}\n")
    
    print("Next steps:")
    print(f"  1. {Colors.YELLOW}Add your API key to .env:{Colors.END}")
    print("     OM_API_KEY=your_key_from_openmind.org")
    print()
    
    print(f"  2. {Colors.YELLOW}Activate virtual environment:{Colors.END}")
    if os_type == "windows":
        print("     .venv\\Scripts\\activate")
    else:
        print("     source .venv/bin/activate")
    print()
    
    print(f"  3. {Colors.YELLOW}Run the Spot agent:{Colors.END}")
    print("     uv run src/run.py spot")
    print()
    
    print(f"  4. {Colors.YELLOW}View WebSim dashboard:{Colors.END}")
    print("     http://localhost:8000/")
    print()
    
    print(f"{Colors.BLUE}For detailed documentation:{Colors.END}")
    print("  - https://docs.openmind.org")
    print("  - GitHub: https://github.com/OpenMind/OM1")
    print()

def print_banner() -> None:
    """Print welcome banner"""
    print(f"""
{Colors.BLUE}{Colors.BOLD}
 ██████╗ ███╗   ███╗ ██╗
██╔═══██╗████╗ ████║ ██║
██║   ██║██╔████╔██║ ██║
██║   ██║██║╚██╔╝██║ ╚═╝
╚██████╔╝██║ ╚═╝ ██║ ██╗
 ╚═════╝ ╚═╝     ╚═╝ ╚═╝
{Colors.END}
""")

def main() -> None:
    """Main setup flow"""
    print_banner()
    
    print_header("OM1 Automated Setup")
    print_info("This script will install OM1 and all dependencies\n")
    
    # Detect OS
    os_type = detect_os()
    print_info(f"Detected OS: {os_type}")
    
    if os_type == "unknown":
        print_error("Unknown operating system")
        sys.exit(1)
    
    # Run setup steps
    try:
        check_python()
        install_uv()
        install_system_deps(os_type)
        setup_repo()
        setup_venv()
        install_dependencies()
        setup_env()
        validate_setup()
        print_next_steps(os_type)
    except KeyboardInterrupt:
        print_warning("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
