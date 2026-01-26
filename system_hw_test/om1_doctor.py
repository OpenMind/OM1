#!/usr/bin/env python3
"""
OM1 Diagnostic Tool
Checks system requirements and configuration for OM1
"""

import sys
import os
import subprocess
from pathlib import Path

# ANSI color codes for pretty output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    """Print section header"""
    print(f"\n{BLUE}{'='*50}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*50}{RESET}\n")

def print_check(name, passed, message=""):
    """Print check result"""
    symbol = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"{symbol} {name}: {status}")
    if message:
        print(f"  → {message}")

def check_python_version():
    """Check if Python version is 3.8+"""
    print_header("Checking Python Version")
    version = sys.version_info
    current = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 8:
        print_check("Python Version", True, f"Python {current}")
        return True
    else:
        print_check("Python Version", False, f"Python {current} (need 3.8+)")
        print(f"  {YELLOW}→ Install Python 3.8 or higher{RESET}")
        return False

def check_requirements():
    """Check if packages from requirements.txt are installed"""
    print_header("Checking Required Packages")
    
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print_check("requirements.txt", False, "File not found")
        return False
    
    print_check("requirements.txt", True, "File found")
    
    # Read requirements
    with open(req_file, 'r') as f:
        packages = [line.strip().split('==')[0].split('>=')[0].split('[')[0] 
                   for line in f if line.strip() and not line.startswith('#')]
    
    missing = []
    for package in packages[:5]:  # Check first 5 packages
        try:
            __import__(package.replace('-', '_'))
            print_check(package, True)
        except ImportError:
            print_check(package, False, "Not installed")
            missing.append(package)
    
    if missing:
        print(f"\n  {YELLOW}→ Install missing packages:{RESET}")
        print(f"    pip install {' '.join(missing)}")
        return False
    
    return True

def check_config_files():
    """Check for config files"""
    print_header("Checking Configuration Files")
    
    config_dir = Path("config")
    if not config_dir.exists():
        print_check("config/ directory", False, "Directory not found")
        return False
    
    print_check("config/ directory", True, "Found")
    
    # Check for any .json5 or .yaml files
    config_files = list(config_dir.glob("*.json5")) + list(config_dir.glob("*.yaml"))
    
    if config_files:
        print_check("Config files", True, f"Found {len(config_files)} files")
        return True
    else:
        print_check("Config files", False, "No config files found")
        return False

def check_gpu():
    """Check for GPU availability (optional)"""
    print_header("Checking GPU (Optional)")
    
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print_check("NVIDIA GPU", True, "GPU detected")
            return True
        else:
            print_check("NVIDIA GPU", False, "No GPU detected (will run in CPU mode)")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print_check("NVIDIA GPU", False, "nvidia-smi not found (will run in CPU mode)")
        return False

def main():
    """Run all diagnostic checks"""
    print(f"\n{BLUE}{'='*50}")
    print("  OM1 Diagnostic Tool")
    print(f"{'='*50}{RESET}\n")
    
    results = {
        "Python Version": check_python_version(),
        "Required Packages": check_requirements(),
        "Config Files": check_config_files(),
        "GPU": check_gpu()
    }
    
    # Summary
    print_header("Summary")
    passed = sum(results.values())
    total = len(results)
    
    print(f"Passed: {passed}/{total} checks")
    
    if passed == total:
        print(f"\n{GREEN}✓ All checks passed! OM1 is ready to run.{RESET}")
        return 0
    elif results["Python Version"] and results["Config Files"]:
        print(f"\n{YELLOW}⚠ Some optional checks failed, but OM1 should work.{RESET}")
        return 0
    else:
        print(f"\n{RED}✗ Critical checks failed. Please fix the issues above.{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())