---
title: TurtleBot4
description: "TurtleBot4 Basic Setup"
icon: robot
---

## Establishing Basic Networking and Connectivity

For many development tasks, you will need to `ssh` to the RPi on the TB4 from your laptop:

```bash
ssh ubuntu@192.168.1.116 # use your TB4's IP address
```

The password is `turtlebot4`. Once on the command line, you can use `turtlebot4-setup` to configure ROS2, the WiFi, etc.

## Required Operating System Versions

OM1 supports TurtleBot4 running ROS2 Humble (image 1.0.4) and Create3 with H.2.6.

### Required Software for the Raspberry Pi

- Ubuntu 22.04.4 LTS (GNU/Linux 5.15.0-1073-raspi aarch64) - combined with TurtleBot4 Humble image
- [turtlebot4_standard_humble_1.0.4.img (2024-08-19 16:59 2.2G)](http://download.ros.org/downloads/turtlebot4/turtlebot4_standard_humble_1.0.4.zip)

Note: choose TurtleBot4 lite image file if your TurtleBot4 is the lite version.

### Required Software for the Create3

[H.2.6 (Humble)](https://github.com/iRobotEducation/create3_docs/releases/download/H.2.6/Create3-H.2.6.swu)

**Important: Make sure your Raspberry Pi 4 is running Ubuntu 22.04.4 LTS (Turtlebot4 Humble image) and that the Create 3 is running H.2.6/Create3-H.2.6.swu**. Among other possible problems, running Ubuntu 24 will create compatibility issues with the version of ROS2 and CycloneDDS on the Create3.

## Prepare the TurtleBot4

1. Flash the correct TurtleBot4 image to an SD card (if needed).
2. Upgrade/downgrade the Create3 firmware to H.2.6 (if needed).
3. Insert the SD card into the TurtleBot4 and power it on.
4. Set up your TurtleBot4 following the [Basic Setup](https://turtlebot.github.io/turtlebot4-user-manual/setup/basic.html#robot).

## Identity and API Keys

**API_KEY** Go to [portal](https://portal.openmind.org/) to get a free API key for the OM1 APIs. Enter this API key in the "api_key" field in the `/config/turtlebot4.json5` file. You can also provide this API key via your `.env` - just enter it as:

```bash
OM_API_KEY=om1_live_e4252f1cf005af...
```

**UNIVERSAL_ROBOT_ID (URID)** Go to "[Hello Robots, come join us](https://portal.openmind.org/robots)" to join a decentralized machine\<\>machine coordination and communication system (FABRIC). Enter machine metadata - currently an arbitrary string - and click "join". The system will provide a unique URID for your robot. The URIDs all share the same format: they begin with `OM`, then 12 alphanumeric characters (numerals and letters), adding up to 14 characters in total. They're not case sensitive. A unique URID allows multiple robots to communicate with one another, similar to how humans use different phone numbers to help them communicate and coordinate.

Enter the URID in the "URID" field in the `/config/turtlebot4.json5` file. You can also provide the URID via your .env - just enter it as:

```bash
URID=OM742d35Cc6634 # yours will be different!
```

Note: for testing you can use any short string as the URID, as long as it's unique for each robot within a team of robots within the same local network.

## Configure the TurtleBot4

Once the TurtleBot4 is set up, configure it as follows:

- Configure the RPi4 through `turtlebot4-setup`
- Configure the Create3 through its web server
- Install docker and configure the `docker-compose.yaml` file

### Configure the TurtleBot4 Internal RPi4

Make the following `ROS_DOMAIN_ID` changes to the TurtleBot4. Use the `turtlebot4-setup` tool to access `ROS Setup:Bash Setup` and set it to the following:

```bash
ROBOT_NAMESPACE=/_your_robot_URID_/pi # example: ROBOT_NAMESPACE=/OM742d35Cc6634/pi
ROS_DOMAIN_ID=0
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
CYCLONEDDS_URI=[] # Empty
```

Click "Save", "Esc", and then "Apply Settings". The TurtleBot4 will reboot.

### Configure the TurtleBot4 Internal Create3

Once the reboot is complete (wait for chime, 1 min), access the Create3's App config page at its web server (e.g. `192.168.1.XXX:8080/ros-config`). Change the ROS_DOMAIN_ID to 1 and enter your robot's URID. The correct settings are:

```
ROS 2 Domain ID (default 0): 1
ROS 2 Namespace: /_your_robot_URID_/c3 # example: /OM742d35Cc6634/c3
RMW_IMPLEMENTATION: rmv_cyclonedds_cpp
```

Basically, you are using the "/c3" prefix to create a unique namespace for the c3. Click `Save` and _Restart Application_. Do not forget to click _Restart Application_, otherwise the changes will not be applied. Wait for chime (1 min) indicating Create3 reboot.

### Install Docker

Finally, on the TurtleBot4's RPi, [install Docker](https://docs.docker.com/engine/install/ubuntu/) and run `sudo docker compose -f docker-compose.yaml up -d`. The `docker-compose` should be:

```bash docker-compose.yaml
services:
  zenoh-bridge-turtlebot4:
    image: openmindagi/turtlebridge
    container_name: zenoh-bridge-turtlebot4
    network_mode: "host"
    restart: always # Ensures the container restarts on reboot
```

The TurtleBot4 will now be more stable, can discover other computers running Zenoh, send them Zenoh messages, and also, accept Zenoh messages and forward them to ROS2.

## OM1 Installation and Launch

You can install and run OM1

- on your laptop, or
- onboard the Raspberry Pi on the TurtleBot4

### Running OM1 on the Internal RPi4

For fully autonomous use, install OM1 on the TurtleBot4's Raspberry Pi. On the RPi terminal command line, follow these [instructions](raspberrypi.md). When you see all the right topics listed in `ros2 topic list`, your TurtleBot4 is set up and you are ready to install OM1.

Install `uv` - Python package manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install `portaudio`, `ffmpeg` and other dependencies:

```bash
sudo apt install pulseaudio pulseaudio-utils ffmpeg portaudio19-dev python-all-dev
```

Then, connect a Logitech 270 Webcam (or equivalent). Unplug the standard TurtleBot4 Depth Camera. Connect a speaker to RPi with a 3.5 mm audio patch cable (or use bluetooth, depending on your pain threshold and patience with debugging bluetooth issues).

Make sure [Identity and Keys](#identity-and-keys) are set correctly. To update the `.env` with your OM1 key, run `vim .env`, "i" to insert, paste in key, "ESC", ":", "wq" to write and exit.

Finally, set Default Input and Output Audio devices. Use `pactl` to set your default microphone and speaker. If you get `pa_context_connect() failed: Connection refused`, then start the audio daemon manually via `pulseaudio --start -D`.

```bash
pactl list sources short                             # List input (microphone) devices
pactl list sinks short                               # List output (speaker) devices

pactl set-default-sink [SINK_NAME || SINK_ID]        # Set default output device
pactl set-default-source [SOURCE_NAME || SOURCE_ID]  # Set default input device
pactl set-sink-volume @DEFAULT_SINK@ 100%            # Set default output volume

# for example,
pactl set-default-source alsa_input.usb-046d_C270_HD_WEBCAM_2CD9A910-02.mono-fallback
pactl set-default-sink alsa_output.platform-bcm2835_audio.stereo-fallback
```

* Launch OM1

```bash
uv run src/run.py turtlebot4_lidar_gps
```

On the RPi4, there will be a lengthy delay before OM1 runs the first time you invoke this command.

### Running OM1 on your Laptop

Install OM1 and Zenoh on your laptop, following [Install OM1 on your laptop](../developing/1_get-started.md) and [Installing the Zenoh router](https://zenoh.io/docs/getting-started/installation/).

Then, run OM1:

```bash
uv run src/run.py turtlebot4
```

Please make sure that your development machine (e.g. your laptop) is running the same version of the Zenoh bridge (e.g. 1.2.1) as the RPi otherwise you will get errors. When you run OM1 on your laptop, OM1 will use your laptop's microphone, speaker, and camera inputs/outputs, rather than the sensors on the TurtleBot4.

### Interacting with TurtleBot4 from a Remote Computer using Zenoh and the Command Line

This is useful for debugging. Inside `system_hw_test`, there are several scripts for you to interact with the TurtleBot4. For all these scripts, provide the robot's URID, such as `OM742d35Cc6634` as an argument.

#### Steer the TurtleBot4 with your Laptop Keyboard

```bash
uv run turtlebot4_keyboard_movement.py --URID OM123435Cc1234
```
* W - Move Forward
* S - Move Backward
* A - Turn Left
* D - Turn Right

#### Read TurtleBot4 Laserscan Data

Provide the robot's URID, such as `OM742d35Cc6634` as an argument:

```bash
uv run rptest.py --URID OM123435Cc1234
```

#### Read TurtleBot4 Battery Data

Provide the robot's URID, such as `OM742d35Cc6634` as an argument:

```bash
uv run turtlebot4_battery.py --URID OM123435Cc1234
```

## Expected ROS2 Topics for a Correctly Configured System

On the TurtleBot4 command line, run `ros2 topic list`. Topics with a `pi` prefix originate from the RPi, and topics with a `c3` prefix are from the Create3. If you do not see any non-prefixed topics, your Create3 is not talking correctly to the RPi.

```bash
/OM742d35Cc6634/c3/battery_state
/OM742d35Cc6634/c3/cmd_vel
/OM742d35Cc6634/c3/dock_status
/OM742d35Cc6634/c3/hazard_detection
/OM742d35Cc6634/c3/imu
/OM742d35Cc6634/c3/odom
/OM742d35Cc6634/c3/robot_state/transition_event
/OM742d35Cc6634/c3/static_transform/transition_event
/OM742d35Cc6634/c3/tf
/OM742d35Cc6634/c3/tf_static

/OM742d35Cc6634/pi/battery_state
/OM742d35Cc6634/pi/cmd_vel
/OM742d35Cc6634/pi/diagnostics
/OM742d35Cc6634/pi/diagnostics_agg
/OM742d35Cc6634/pi/diagnostics_toplevel_state
/OM742d35Cc6634/pi/dock_status
/OM742d35Cc6634/pi/function_calls
/OM742d35Cc6634/pi/hazard_detection
/OM742d35Cc6634/pi/hmi/buttons
/OM742d35Cc6634/pi/hmi/display
/OM742d35Cc6634/pi/hmi/display/message
/OM742d35Cc6634/pi/hmi/led
/OM742d35Cc6634/pi/imu
/OM742d35Cc6634/pi/interface_buttons
/OM742d35Cc6634/pi/ip
/OM742d35Cc6634/pi/joint_states
/OM742d35Cc6634/pi/joy
/OM742d35Cc6634/pi/joy/set_feedback
/OM742d35Cc6634/pi/mouse
/OM742d35Cc6634/pi/oakd/imu/data
/OM742d35Cc6634/pi/oakd/rgb/preview/camera_info
/OM742d35Cc6634/pi/oakd/rgb/preview/image_raw
/OM742d35Cc6634/pi/oakd/rgb/preview/image_raw/compressed
/OM742d35Cc6634/pi/oakd/rgb/preview/image_raw/compressedDepth
/OM742d35Cc6634/pi/oakd/rgb/preview/image_raw/theora
/OM742d35Cc6634/pi/robot_description
/OM742d35Cc6634/pi/scan
/OM742d35Cc6634/pi/tf
/OM742d35Cc6634/pi/tf_static
/OM742d35Cc6634/pi/wheel_status

/diagnostics
/parameter_events
/rosout
```

## Building the Docker Dual Bridge Images

You can build your own dual bridge docker images using the provided `Dockerfile` (see `/system_hw_test/turtlebot_zenoh/Dockerfile`):

```bash
docker build -t my-username/my-image .
docker push my-username/my-image
```

Useful docker commands:

```bash
sudo docker compose pull # pull latest image

# Images
sudo docker images
sudo docker rmi IMAGE_ID --force

# Containers
sudo docker ps -a
sudo docker attach CONTAINER_ID # to stream logs
sudo docker exec -it CONTAINER_ID sh # to get a shell
sudo docker kill CONTAINER_ID
```

## Debugging Commands

Webcam debugging:

```bash
lsusb
sudo apt install v4l-utils
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --stream-mmap --all
```
