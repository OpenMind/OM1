---
title: System Setup
description: "System Setup"
icon: gear
---

This guide walks you through setting up a fresh Nvidia Thor system running JetPack 7.0, to run OM1. Follow each section in order and verify installations as you go.

## Quick Checklist

- Python 3.10+ installed
- `uv` package manager
- Docker
- Docker Compose
- Audio support (PyAudio + PortAudio)
- FFmpeg for video processing
- Chrome browser
- ROS2
- CycloneDDS

### uv
Use `curl` to download and install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Docker

Docker is pre-installed on JetPack 7.0 systems, but you need to configure permissions:

```bash
newgrp docker

# Add current user to docker group
sudo usermod -aG docker $USER

# Verify group membership
groups
```

You should see `docker` in the list of groups. If not, log out and log back in, then check again.

### docker-compose

Download and install Docker Compose with the following commands:

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.34.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
```

Make it executable:

```bash
sudo chmod +x /usr/local/bin/docker-compose
```

Verify the installation:

```bash
docker-compose --version
```

### Poetry (Optional)

Install Poetry using the official installation script:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install poetry shell for the environment management:

```bash
poetry self add poetry-plugin-shell
```

### Pyaudio (For microphone support)

Required for microphone input and audio processing:

```bash
sudo apt install portaudio19-dev python3-pyaudio
```

### FFmpeg (For video processing)

Install FFmpeg using the following command:

```bash
sudo apt install ffmpeg
```

### Chrome (For web interface)

Download and install Google Chrome:

```bash
sudo snap install chromium
```

Hold snap updates to prevent automatic updates:

```bash
snap download snapd --revision=24724
sudo snap ack snapd_24724.assert
sudo snap install snapd_24724.snap
sudo snap refresh --hold snapd
```

### ROS2 (Optional)

Follow the official ROS2 installation guide for Ubuntu: [ROS2 Installation](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).

After installing ROS2, source the ROS2 setup script:

```bash
source /opt/ros/jazzy/setup.bash
```

You can add this line to your ~/.bashrc file to source it automatically on terminal startup.

### CycloneDDS Binary (Optional)

Install CycloneDDS for ROS2 communication:

```bash
sudo apt install ros-jazzy-rmw-cyclonedds-cpp
sudo apt install ros-jazzy-rosidl-generator-dds-idl
```

Now, set CycloneDDS as the default RMW implementation by adding the following line to your ~/.bashrc file:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

You can restart your ROS2 daemon with the following command:

```bash
ros2 daemon stop
ros2 daemon start
```

### CycloneDDS Build from Source (Optional)

If you prefer to build CycloneDDS from source, use the following commands:

```bash
cd Documents
mkdir -p GitHub && cd GitHub
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=ON
cmake --build . --target install
```

Then you need to set the following environment variables in your ~/.bashrc file:

```bash
export CYCLONEDDS_HOME=$HOME/Documents/GitHub/cyclonedds/install
```

### Configure Network Settings (Unitree Only)

You need to open the network settings and find the network interface that the robot connected. In IPv4 settings, set the method to Manual and add the following IP address:

`192.168.123.xxx`

and set the subnet mask to

`255.255.255.0`

### CycloneDDS Configuration (Optional)

You can create a CycloneDDS configuration file to customize its behavior. Create a file named cyclonedds.xml in your home directory:

```bash
<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface name="enP2p1s0" priority="default" multicast="default" />
      </Interfaces>
    </General>
    <Discovery>
      <EnableTopicDiscoveryEndpoints>true</EnableTopicDiscoveryEndpoints>
    </Discovery>
  </Domain>
</CycloneDDS>
```

Then, set the CYCLONEDDS_URI environment variable in your ~/.bashrc file:

```bash
export CYCLONEDDS_URI=file://$HOME/cyclonedds.xml
```

### v4l2-ctl (For camera configuration)

Install v4l2-ctl using the following command:

```bash
sudo apt install v4l-utils
```

For next steps, you will need the BrainPack. If you don't have the BrainPack yet, you can skip this.

Follow the guide [here](./system_config.md), to proceed.
