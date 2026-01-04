# Running OM1 on Windows using WSL2

OM1 officially targets Linux (Ubuntu 20.04, 22.04, or 24.04).  
Windows users can run OM1 reliably using **WSL2 (Windows Subsystem for Linux)**.

This guide documents a tested setup using **Windows 11 + WSL2 + Ubuntu 22.04**.

---

## Prerequisites

- Windows 10 (21H2+) or Windows 11
- WSL2 enabled
- Ubuntu 22.04 LTS (recommended)
- Python 3.10+

---

## Install WSL2 and Ubuntu

Open **PowerShell (Admin)** and run:

```powershell
wsl --install

