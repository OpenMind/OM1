# OM1 Windows (WSL) Setup Guide

This guide explains how to run OM1 on Windows using WSL (Windows Subsystem for Linux).
No prior Linux experience is required.

---

## 1. Requirements

- Windows 10 or Windows 11
- Internet connection
- Microphone (for ASR features)

---

## 2. Install WSL

Open PowerShell as Administrator and run:

wsl --install

Restart your computer when prompted.

After restart, open Ubuntu from the Start Menu and complete the setup.

---

## 3. Install Docker Desktop

1. Download Docker Desktop for Windows: https://www.docker.com/products/docker-desktop
2. During installation, enable:
   - "Use WSL 2 backend"
3. Open Docker Desktop
4. Go to Settings → Resources → WSL Integration
5. Enable Ubuntu

---

## 4. Clone OM1 Repository

Inside Ubuntu (WSL), run:

git clone https://github.com/OpenMind/OM1.git
cd OM1

---

## 5. Install OM1

pip install -e .

---

## 6. Audio & Microphone Notes

If audio input is not detected:
- Make sure microphone permission is enabled in Windows
- Restart WSL after changing audio settings

Some ASR features may not work perfectly on WSL. This is a known limitation.

---

## 7. Test Installation

Run:

om1 run spot

If the agent starts, the setup is complete.

---

## 8. Common Issues

- NO AVAILABLE INPUTS: Check microphone permissions
- Docker not found: Ensure Docker Desktop is running
- Permission errors: Restart WSL and try again
