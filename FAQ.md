# Frequently Asked Questions (FAQ)

Common questions about OM1 setup, configuration, and troubleshooting.

## Installation

### Q: Do I need a real robot to use OM1?
**A:** No! You can run OM1 in simulation mode or just test agents without hardware.

### Q: What operating systems are supported?
**A:** Linux (Ubuntu 20/22/24) and macOS 12.0+. Windows is not officially supported but may work.

### Q: I get "uv: command not found" - what do I do?
**A:** Install uv package manager:
- Mac: `brew install uv`
- Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Q: I get "portaudio not found" errors
**A:** Install portaudio:
- Mac: `brew install portaudio`
- Linux: `sudo apt-get install portaudio19-dev python3-all-dev`

## API Key

### Q: Where do I get an API key?
**A:** Get your free API key at https://portal.openmind.org/

### Q: Can I use "openmind_free" as my API key?
**A:** No! That's just a placeholder. You must replace it with your real key from the portal.

### Q: Where do I put my API key?
**A:** Two options:
1. In `/config/spot.json5`: Replace `"openmind_free"` with your key
2. Create `.env` file: Add `OM_API_KEY=your_key_here`

### Q: My API key isn't working
**A:** Check:
- Key starts with `om1_live_`
- No extra spaces or quotes
- You replaced the placeholder, not added to it
- Restart terminal after setting .env

## Running OM1

### Q: How do I start an agent?
**A:** `uv run src/run.py spot` (or replace `spot` with your agent name)

### Q: Can I use python instead of uv?
**A:** Yes, but uv is recommended. Both work: `uv run src/run.py spot` OR `python src/run.py start spot`

### Q: WebSim won't load at localhost:8000
**A:** 
- Wait 30-60 seconds after starting
- Check firewall isn't blocking port 8000
- Try http://127.0.0.1:8000 instead

### Q: Agent starts but nothing happens
**A:** Check that ASR (speech) and TTS (text-to-speech) are configured in your agent config file.

## Docker

### Q: Do I need Docker?
**A:** Only for full autonomy features or if you want to run services in containers.

### Q: Docker permission denied on Linux
**A:** Run: `sudo usermod -aG docker $USER` then logout and login again

## Hardware

### Q: What robots are supported?
**A:** Unitree Go2, Unitree G1, TurtleBot4, UbTech Yanshee, and custom robots via plugins.

### Q: Can I connect my own robot?
**A:** Yes! OM1 supports custom hardware via Zenoh, ROS2, CycloneDDS, or custom APIs.

### Q: Do I need cameras/sensors?
**A:** Not required for basic testing, but needed for vision/sensor-based agents.

## Configuration

### Q: How do I create a custom agent?
**A:** Copy an existing config file in `/config/`, modify it, then run with your new config name.

### Q: Can I use different LLM providers?
**A:** Yes! OM1 supports OpenAI, Gemini, DeepSeek, xAI, Anthropic, and more.

### Q: How do I switch between agents?
**A:** Run: `uv run src/run.py [agent_name]` - OM1 remembers the last agent used.

## Troubleshooting

### Q: "No module named X" error
**A:** Run `uv sync` to install all dependencies.

### Q: Changes to code not taking effect
**A:** 
1. Stop the agent completely
2. Clear cache: `find . -type d -name "__pycache__" -exec rm -rf {} +`
3. Restart

### Q: High memory usage
**A:** Reduce model sizes in config or close other applications.

## Contributing

### Q: How can I contribute?
**A:** See [CONTRIBUTING.md](CONTRIBUTING.md) - contributions welcome!

### Q: I found a bug, what should I do?
**A:** Open an issue on GitHub with details: OS, Python version, error message, steps to reproduce.

### Q: Can I contribute from mobile?
**A:** Yes! Documentation improvements, bug reports, and simple PRs can be done from mobile.

## Getting Help

### Q: Where can I get more help?
**A:** 
- Documentation: https://docs.openmind.org
- GitHub Issues: https://github.com/OpenMind/OM1/issues
- Discord/Telegram community
- GitHub Discussions

### Q: How do I report a security issue?
**A:** See [SECURITY.md](SECURITY.md) or contact the team privately.

---

**Don't see your question?** Check the [Troubleshooting Guide](TROUBLESHOOTING.md) or open a GitHub issue!
