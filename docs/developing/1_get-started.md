# Getting Started

This guide walks you through running your first OpenMind OM1 instance on a local machine.

## Prerequisites
- Python 3.10+
- Git
- An OpenMind API key from https://portal.openmind.org

## Quick start
```bash
git clone https://github.com/OpenMind/OM1.git
cd OM1
uv venv
source .venv/bin/activate
export OPENMIND_API_KEY=your_api_key_here
uv run src/run.py spot

