---
title: Nvidia Thor Dev Kit
description: "Getting Started with OM1 on Nvidia Thor Dev Kit"
---

## Basics

Make sure you have the following before installation:

- Device: 945-14070-0080-000 | Jetson AGX Thor Developer Kit
- USB Memory Stick

Install the latest Dev Kit base image [following this guide]
(https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html).

Helpful hint: If you are on a mac, you can quickly wipe the USB drive for use with Balena Etcher with these commands:

```bash
diskutil list
diskutil unmountDisk /dev/disk4 # customize this, obviously
diskutil eraseDisk FAT32 SANDISK /dev/disk4
```

We typically use the `Monitor-attached` flow. We have successfully used `jetsoninstaller-0.2.0-r38.2-2025-08-22-01-33-29-arm64.iso`. Sometimes the installation hangs right after `freeing initrd memory` at t = 4s, but just restart the Thor and try again; it should work eventually. You will be left with the Thor running Ubuntu 24.04.3 LTS.

## JetPack SDK

Once you have installed Ubuntu, [install JetPack](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_jetpack.html).

## System Basics

```bash
sudo apt-get update

# browser
sudo apt install chromium-browser

# gihub desktop

# directly download from github. Use the most recent release for arm64, such as
# https://github.com/shiftkey/desktop/releases/download/release-3.4.13-linux1/GitHubDesktop-linux-arm64-3.4.13-linux1.deb
# install by double-clicking the .deb

# sublime
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo tee /etc/apt/keyrings/sublimehq-pub.asc > /dev/null
echo -e 'Types: deb\nURIs: https://download.sublimetext.com/\nSuites: apt/stable/\nSigned-By: /etc/apt/keyrings/sublimehq-pub.asc' | sudo tee /etc/apt/sources.list.d/sublime-text.sources
sudo apt-get update
sudo apt-get install sublime-text

# camera device info
v4ls-ctl --listdevices

# to test cameras
sudo apt install guvcview # FYI the cheese webcam app is broken on Thor
```

You can run the usual Ubuntu software updater without hosing the install.

## System Info

* **GPU**: You can get GPU stats and other basic information via the built in command `nvidia-smi`. The command should show CUDA version 13.

* **tegastats**: The `tegrastats` command (via the top right corner Nvidia dropdown) gives you information about system temperatures, frequencies, and power consumption in a terminal.

* **Jetson Power GUI** gives you a visual overview of CPU, GPU, Thermal, Power, and fans.

## OM1

Then, install OM1 on the Nvidia Thor from its command line, following the [standard instructions](../developing/1_get-started.mdx).

## RealSense Depth Camera

* Install `cmake`:

```bash
sudo apt install cmake
```

Download the `development` branch of [librealsense](https://github.com/IntelRealSense/librealsense/tree/development).
Follow [Building from Source using Native Backend](https://github.com/IntelRealSense/librealsense/blob/development/doc/installation_jetson.md#building-from-source-using-native-backend).

* Patch `./scripts/patch-realsense-ubuntu-L4T.sh`

Add `38.2.2` to the list of supported versions:

```bash
Line 100 "38.2") -> "38.2" | "38.2.2")
```

* Make sure your local CUDA is working; test with the [Nvidia examples](https://github.com/NVIDIA/cuda-samples):

```bash
python3 run_tests.py --output ./test --dir ./build/Samples --config test_args.json
```

* Add needed env vars to your `.bashrc`:

```bash
echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export CUDACXX=/usr/local/cuda-13.0/bin/nvcc" >> ~/.bashrc
source ~/.bashrc
```

* Install OpenGL Utility Library (GLU)

```bash
sudo apt-get install libglu1-mesa-dev
```

* Change the `cmake` command relative to "building-from-source-using-native-backend"

Specifically, `-DCMAKE_CUDA_ARCHITECTURES=native` seems to be needed. The resulting command is:

```bash
cmake .. -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=release -DFORCE_RSUSB_BACKEND=false -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_WITH_CUDA=true && make -j$(($(nproc)-1)) && sudo make install
```

Finally, plug in your RealSense. When you now run `realsense-viewer`, it _should_ just work.

**Warning** The RealSense camera requires the full bandwidth of an entire USB 3.2 hub. This means that on the Thor, depending on where you plug in the RealSense, the adjacent USB port will stop working (practically speaking). So e.g. you cannot use one USB 3.2 A port for your mouse, and the other for the DepthSense. We suggest adding a hub to one of the USB Type-C ports, and then using that hub for your mouse, keyboard, microphones, speakers, and other cameras, etc, and then the RealSense can use/block both USB 3.2 Type A ports.

## Robot Deployments

For robot deployments, we recommend modules such as the [Auvidea Jetson T5000 module](https://auvidea.eu/press-release-nvidia-jetson-thor-t5000-launch/) since it directly accepts 24V to 48V, unlike the dev kit's more restricted 9–28V DC Micro-fit power connector.

## Done

You are now running OM1 on your Nvidia Thor. Explore its capabilities and refer to the [OM1 GitHub repository](https://github.com/openmind/OM1) for advanced settings and next steps.
