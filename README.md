### Beginner Tips for Setup (Based on Common Issues)

If you're new to OM1 and setting up on Windows with WSL (Ubuntu), here are some tips from my experience to avoid common problems:

- **WSL Network Issues** (e.g., "apt update" fails with "Network is unreachable"): In Windows PowerShell, run `wsl --shutdown`, then restart your terminal. This resets the network.
- **Build Errors During `uv sync`** (e.g., "command 'cc' failed" or "portaudio.h: No such file or directory"): Install missing packages with `sudo apt install build-essential portaudio19-dev ffmpeg git python3-dev -y`.
- **Config Syntax Errors** (e.g., "Unexpected '"' at column 29" in spot.json5): Use unquoted keys like `api_key: your_key_here`, and add commas after each line except the last. Avoid extra quotes.
- **UI Interaction Issues** (WebSim at http://localhost:8000/): The UI is for monitoring states (e.g., "wag tail", "excited"). Use voice commands (allow mic permissions in browser/Windows) like "Hello Spot" or "What do you see?" to trigger responses. No text input box by default—voice activates the LLM and deducts credits.
- **Credits Not Deducting**: This happens if no API calls are triggered (e.g., no interactions). After voice commands, check the portal—usage shows after 1-2 responses.
- **Blockchain 503 Error** (e.g., "Loading rules from Ethereum blockchain failed with 503"): Often temporary due to server overload. Wait 10-15 mins and retry. Doesn't affect local virtual runs.

These can save hours for non-technical users! For more, see the full docs at https://docs.openmind.org/developing/1_get-started.