---
title: ROS2-humble
description: "Installation"
icon: gear
---

## Install ROS2-humble

### Setup Sources

You will need to add the ROS 2 apt repository to your system.

First ensure that the Ubuntu Universe repository is enabled.

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
```

The ros-apt-source packages provide keys and apt source configuration for the various ROS repositories.

Installing the `ros2-apt-source` package will configure ROS 2 repositories for your system. Updates to repository configuration will occur automatically when new versions of this package are released to the ROS repositories.

```bash
sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
```

Now, run

```bash
sudo apt update
sudo apt upgrade
sudo apt install ros-humble-desktop
```

This will install ROS and all the relevant packages.

> **Note:** When installing ros2-humble for Gazebo, run `sudo apt install ros-humble-desktop-full` instead of sudo apt install `ros-humble-desktop`. It will install ROS, RViz, and Gazebo.

Refer [ros2-humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html), to get a better understanding.
