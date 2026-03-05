---
title: Middleware Setup
description: "ROS2 and DDS Setup"
icon: gear
---


## Install zenoh-bridge

Add Eclipse Zenoh private repository to the sources list:

```bash
curl -L https://download.eclipse.org/zenoh/debian-repo/zenoh-public-key | sudo gpg --dearmor --yes --output /etc/apt/keyrings/zenoh-public-key.gpg
echo "deb [signed-by=/etc/apt/keyrings/zenoh-public-key.gpg] https://download.eclipse.org/zenoh/debian-repo/ /" | sudo tee -a /etc/apt/sources.list > /dev/null
sudo apt update
```

install the standalone executable with: `sudo apt install zenoh-bridge-ros2dds`.
