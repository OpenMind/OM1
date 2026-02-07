---
title: OTA Setup
description: "Setup OTA"
icon: rotate
---

## Cloud Docker Management Service

The cloud docker management service allows remote management of Docker containers via a web interface. To enable this service, follow these steps:

### Step 1: Sign up

Sign up for an account on [OpenMind Portal](https://portal.openmind.org/).

### Step 2: Get OpenMind API key

Get you API key from the [Dashboard](https://portal.openmind.org/) page.

### Step 3: Set the API key

Set your API key as an environment variable in your Bash profile:

```bash
vim ~/.bashrc
```

Add the following lines:

```bash
export OM_API_KEY="your_api_key_here"
```

### Step 4: Get the API Key ID

Get API_KEY_ID from the [Dashboard](https://portal.openmind.org/) page. The API Key ID is a 16-digit character string, such as `om1_live_<16 characters>`. Now, export the API Key ID as an environment variable:

```bash
vim ~/.bashrc
```

```bash
export OM_API_KEY_ID="your_api_key_id_here"
```

Reload your Bash profile:

```bash
source ~/.bashrc
```

### Step 5: Set the robot type that you are using

```bash
vim ~/.bashrc

export ROBOT_TYPE="go2"  # or "go1", "tron"
```

### Setup OTA Update Services

To enable the Over-The-Air (OTA) update service for Docker containers, you need to set up two docker services: `ota_agent` and `ota_updater`. These services will allow you to manage and update your Docker containers remotely via the OpenMind Portal.

To create a `ota_updater.yml` file, follow these steps:

```bash
cd ~
vim ota_updater.yml
```

Copy the content from the [ota_updater.yml template](https://github.com/OpenMind/OM1-deployment/blob/main/latest/ota_updater.yml) to `ota_updater.yml` file.

> **Note:** You can use the stable version as well. The file example provided is the latest version.

Save and close the file (`:wq` in vim).

Start OTA Updater Service

```bash
docker-compose -f ota_updater.yml up -d
```

A `.ota` directory will be automatically created in your home directory to store OTA configuration files.

Now, you can set up the `ota_agent` service. Create an `ota_agent.yml` file:

Navigate to the OTA directory:

```bash
cd ~/.ota
vim ota_agent.yml
```

Copy the content from the [ota_agent.yml template](https://github.com/OpenMind/OM1-deployment/blob/main/latest/ota_agent.yml) to `ota_agent.yml` file.

> **Note:** You can use the stable version as well. The file example provided is the latest version.

Save and close the file.

Start OTA Agent Service

```bash
docker-compose -f ota_agent.yml up -d
```

**Verify both services are running:**

```bash
docker ps | grep ota_updater
docker ps | grep ota_agent
```

Expected output: Both `ota_updater` and `ota_agent` containers listed.

You can now manage and update your Docker containers remotely via the [OpenMind Portal](https://portal.openmind.org/).

### Model Downloads

### Riva Models

Riva models are encrypted and require authentication to download. To download Riva models, you need to set up the NVIDIA NGC CLI tool.

#### Install NGC CLI

> **⚠️ Warning:** Run the following commands in your root directory (`cd ~`). Otherwise, Docker Compose may not locate the required files.

To generate your own NGC api key, check this [video](https://www.youtube.com/watch?v=yBNt4qSnn0k).

```bash
wget --content-disposition https://ngc.nvidia.com/downloads/ngccli_arm64.zip && unzip ngccli_arm64.zip && chmod u+x ngc-cli/ngc
find ngc-cli/ -type f -exec md5sum {} + | LC_ALL=C sort | md5sum -c ngc-cli.md5
echo export PATH=\"\$PATH:$(pwd)/ngc-cli\" >> ~/.bash_profile
source ~/.bash_profile
ngc config set
```

This will ask several questions during the install. Choose these values:

```
Enter API key [no-apikey]. Choices: [<VALID_APIKEY>, 'no-apikey']: <YOUR_API_KEY>
Enter CLI output format type [ascii]. Choices: ['ascii', 'csv', 'json']: ascii
Enter org [no-org]. Choices: ['<YOUR_ORG>']: <YOUR_ORG>
Enter team [no-team]. Choices: ['<YOUR_TEAM>', 'no-team']: <YOUR_TEAM>
Enter ace [no-ace]. Choices: ['no-ace']: no-ace
```

> **⚠️ Warning:** NGC CLI creates a `.bash_profile` file if it doesn't exist. If you already have a `.bashrc` file, merge them manually to avoid losing your bash configuration.

ngc cli will create a .bash_profile file if it does not exist. If you already have a .bashrc file, please make sure to merge the two files properly. Otherwise, your bash environment may not work as expected.

#### Download Riva Models

Download Riva Embedded version models for Jetson 7.0:

```bash
ngc registry resource download-version nvidia/riva/riva_quickstart_arm64:2.24.0
```

```bash
cd riva_quickstart_arm64_v2.24.0
sudo bash riva_init.sh
```

This will ask the NGC api key to download the model, use `<YOUR_API_KEY>`. It will take a while to download.

> **Note:** The following command is for testing.

Run Riva locally:

```bash
cd riva_quickstart_arm64_v2.24.0
bash riva_start.sh
```
Now, please expose these environment variables in your ~/.bashrc file to use Riva service:

```bash
export RIVA_API_KEY=<YOUR_API_KEY>
export RIVA_API_NGC_ORG=<YOUR_ORG>
export RIVA_EULA=accept
```

```bash
source ~/.bashrc
```

### OpenMind Riva Docker Image for Jetson

We created a `openmindagi/riva-speech-server:2.24.0-l4t-aarch64` docker image that has Riva ASR and TTS endpoints with example code to run Riva services on Jetson devices. You can pull the image directly without downloading the models from NGC:

```bash
docker pull openmindagi/riva-speech-server:2.24.0-l4t-aarch64
```

The dockerfile can be found [here](https://github.com/OpenMind/OM1-deployment/blob/main/docker/riva/Dockerfile) and the docker-compose file can be found [here](https://github.com/OpenMind/OM1-deployment/blob/main/docker-compose/riva-compose.yml).

> **Note:** Once you download the models from NGC and export the environment variables, you can use OpenMind Portal to download Riva dockerfile and run Riva services.

### Test Riva Services

Once you have Riva services running, you can use the following script to test the ASR and TTS endpoints:

```bash
git clone https://github.com/OpenMind/OM1-modules.git

cd OM1-modules

# Activate poetry shell
poetry shell

# Install dependencies
poetry install

# Test ASR
python3 -m om1_speech.main --remote-url=ws://localhost:6790

# Test TTS
poetry run om1_tts --tts-url=https://api-dev.openmind.org/api/core/tts --device=<optional> --rate=<optional>
```

## Port Reference

Services use the following ports:

| Port | Service | Purpose |
|------|---------|---------|
| 1935 | MediaMTX RTMP Server | Video streaming |
| 6790 | Riva ASR WebSocket | Speech recognition API |
| 6791 | Riva TTS HTTP | Text-to-speech API |
| 8000 | MediaMTX RTMP API | RTMP control |
| 8001 | MediaMTX HLS API | HLS streaming |
| 8554 | MediaMTX RTSP | RTSP streaming |
| 8860 | Qwen 30B Quantized | LLM inference |
| 8880 | Kokoro TTS | TTS engine |
| 8888 | MediaMTX Streaming | Streaming control |
| 50000 | Riva Server API | Internal Riva API |
| 50051 | Riva NMT Remote API | Remote TTS/ASR APIs |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **OTA services won't start** | Check API key is correct: `echo $OM_API_KEY` |
| **NGC CLI not found** | Verify PATH: `echo $PATH` includes ngc-cli directory |
| **Riva models download fails** | Confirm NGC API key is valid and you have quota |
| **Port already in use** | Check what's running: `sudo lsof -i :PORT_NUMBER` |
| **Docker permission denied** | Add user to docker group: `sudo usermod -aG docker $USER` |
