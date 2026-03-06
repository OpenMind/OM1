---
title: Zenoh Bridge
description: "Installation"
icon: gear
---

ROS (the Robot Operating System) is a set of software libraries and tools allowing to build robotic applications. In its version 2, ROS 2 relies mostly on O.M.G. DDS as a middleware for communications. This plugin bridges all ROS 2 communications using DDS over Zenoh.

While a Zenoh bridge for DDS already exists and helped lot of robotic use cases to overcome some wireless connectivity, bandwidth and integration issues, using a bridge dedicated to ROS 2 brings the following advantages:

A better integration of the ROS graph (all ROS topics/services/actions can be seen across bridges)
A better support of ROS toolings (ros2, rviz2...)
Configuration of a ROS namespace on the bridge, instead of on each ROS Nodes
Easier integration with Zenoh native applications (services and actions are mapped to Zenoh Queryables)
More compact exchanges of discovery information between the bridges

## Install zenoh-bridge

Add Eclipse Zenoh private repository to the sources list:

```bash
curl -L https://download.eclipse.org/zenoh/debian-repo/zenoh-public-key | sudo gpg --dearmor --yes --output /etc/apt/keyrings/zenoh-public-key.gpg
echo "deb [signed-by=/etc/apt/keyrings/zenoh-public-key.gpg] https://download.eclipse.org/zenoh/debian-repo/ /" | sudo tee -a /etc/apt/sources.list > /dev/null
sudo apt update
```

Now you can install the standalone executable with: `sudo apt install zenoh-bridge-ros2dds`.
