---
title: Installation Guide
description: "Learn how to install, set up and configure OM1."
---

## System Requirements

### Operating System

- Linux (Ubuntu 20, 22, 24)
- MacOS 12.0+

### Hardware

- Sufficient memory to run vision and other models
- Reliable WiFi or other networking
- Sensors such as cameras, microphones, LIDAR units, IMUs
- Actuators and outputs such as speakers, visual displays, and movement platforms (legs, arms, hands)
- Hardware connected to the "central" computer via `Zenoh`, `CycloneDDS`, serial, usb, or custom APIs/libraries

### Software

Ensure you have the following installed on your machine:

- `Python` >= 3.10
- `uv` >= 0.6.2 as the Python package manager and virtual environment
- `portaudio` for audio input and output
- `ffmpeg` for video processing
- Get your OpenMind API key [here](https://portal.openmind.org/)

#### UV (A Rust and Python package manager)

```bash
# Mac
brew install uv

# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### PortAudio Library

For audio functionality, install `portaudio`:

```bash
# Mac
brew install portaudio

# Linux
sudo apt-get update
sudo apt-get install portaudio19-dev
```

#### Install python3-dev

```bash
# Linux
sudo apt-get update
sudo apt-get install python3-dev
```

#### ffmpeg

For video functionality, install FFmpeg:

```bash
# Mac
brew install ffmpeg

# Linux
sudo apt-get update
sudo apt-get install ffmpeg
```
## CLI

OM1 provides a command-line interface (CLI). The main entry point is `src/run.py` which provides the following commands:

- `start`: Start an agent with a specified config

```bash
python src/run.py start [config_name] [--log-level] [--log-to-file]
```

- `config_name`: Name of the config file (without `.json5` extension) in the `/config` directory.
- `--log-level`: Optional log level (default: `INFO`). Use `DEBUG` for detailed logs.
- `--log-to-file`: Optional flag to log to `logs/{config_name}.log` (default: `False`).

## Installation and Setup

1. Clone the repository

Run the following commands to clone the repository and set up the environment:

```bash clone repo
git clone https://github.com/OpenMind/OM1.git
cd OM1
git submodule update --init
uv venv
```

2. Set the configuration variables

Locate the `config` folder and add your OpenMind API key to `/config/spot.json5` (for example). If you do not already have one, you can obtain a free access key at https://portal.openmind.org/.

```bash
# /config/spot.json5
...
"api_key": "om1_live_..."
...
```

Or, create a `.env` file in the project directory and add the following:

Note: Using the placeholder key **openmind_free** will generate errors.

```bash
OM_API_KEY=om1_live_...
```

3. Run the Spot Agent

Run the following command to start the Spot Agent:

```bash
uv run src/run.py spot
```

**Note**: Agent configuration names are only required when switching between different agents. Once an agent has been run, it becomes the default for subsequent executions.

Spot is just an example agent configuration.

If you want to interact with the agent and see how it works, make sure ASR and TTS are configured in `spot.json5`.

ASR configuration (check in agent_inputs)
```bash
{
      "type": "GoogleASRInput"
}
```

TTS configuration (check in agent_actions)
```bash
{
      name: "speak",
      llm_label: "speak",
      connector: "elevenlabs_tts",
      config:
      {
        voice_id: "i4CzbCVWoqvD0P1QJCUL",
        "silence_rate": 20,
      },
}
```

During the first execution, the system will automatically resolve and install all project dependencies. This process may take several minutes to complete before the agent becomes operational.

**Runtime Configuration**

Upon successful initialization, a `.runtime.json5` file will be generated in the `config/memory` directory. This file serves as a snapshot of the agent configuration used in the current session.

**Subsequent Executions**

After the initial run, you can start the agent using the simplified command:

```bash
uv run src/run.py
```

![Reference](../assets/hot_reload.png)

The system will automatically load the most recent agent configuration from memory. Additionally, a `.runtime.json5` file will be created in the root config directory, which persists across sessions unless a different agent configuration is specified.

**Switching Agent Configurations**

To run a different agent (for example, the conversation agent), specify the configuration name explicitly:

```bash
uv run src/run.py conversation
```

### WebSim to check input and output

Go to [http://localhost:8000](http://localhost:8000) to see real time logs along with the input and output in the terminal. For easy debugging, add `--debug` to see additional logging information.

### Understanding the Log Data

The log data provide insight into how the `spot` agent makes sense of its environment and decides on its next actions.

  - First, it detects a person using vision.
  - Communicates with an external AI API for response generation.
  - The LLM(s) decide on a set of actions (dancing and speaking).
  - The simulated robot expresses emotions via a front-facing display.
  - Logs latency and processing times to monitor system performance.

```bash
Object Detector INPUT
// START
You see a person in front of you. You also see a laptop.
// END

AVAILABLE ACTIONS:
command: move
    A movement to be performed by the agent.
    Effect: Allows the agent to move.
    Arguments: Allowed values: 'stand still', 'sit', 'dance', 'shake paw', 'walk', 'walk back', 'run', 'jump', 'wag tail'

command: speak
    Words to be spoken by the agent.
    Effect: Allows the agent to speak.
    Arguments: <class 'str'>

command: emotion
    A facial expression to be performed by the agent.
    Effect: Performs a given facial expression.
    Arguments: Allowed values: 'cry', 'smile', 'frown', 'think', 'joy'

What will you do? Command:

INFO:httpx:HTTP Request: POST https://api.openmind.org/api/core/openai/chat/completions "HTTP/1.1 200 OK"
INFO:root:OpenAI LLM output: commands=[Command(type='move', value='wag tail'), Command(type='speak', value="Hi there! I see you and I'm excited!"), Command(type='emotion', value='joy')]
```

## More Examples

There are more pre-configured agents in the `/config` folder. They can be run with the following command:

For example, to run the `cubly` agent:

```bash
uv run src/run.py cubly
```

If you configure a custom agent, replace `<agent_name>` with your agent and run the below command:

```bash
uv run src/run.py <agent_name>
```

To get started with development, refer [here](https://docs.openmind.org/developer_cookbook/introduction)
