# Windows WSL Setup Guide for OM1 🤖

## Why WSL?
OM1 requires Ubuntu 22.04. Windows users MUST use WSL2.

## Prerequisites
- Windows 10 (build 19041+) OR Windows 11
- 16GB RAM recommended
- NVIDIA GPU (optional)

## 🚀 Installation Steps

### Step 1: Install WSL2
**PowerShell as Administrator:**
```powershell
wsl --install
wsl --install -d Ubuntu-22.04
