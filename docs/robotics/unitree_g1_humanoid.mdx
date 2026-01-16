---
title: Unitree G1 Humanoid
description: "Unitree G1 Humanoid"
---

## Useful links and Reading

https://support.unitree.com/home/en/G1_developer
https://github.com/unitreerobotics/avp_teleoperate

C++ SDK
https://github.com/unitreerobotics/unitree_sdk2

Python SDK
https://github.com/unitreerobotics/unitree_sdk2_python

## Basic Command

Run

```bash Run Robot
uv run src/run.py unitree_g1_humanoid
```

### Installation on Mac

```bash
brew install uv portaudio cmake
```

Install cycloneDDS via https://cyclonedds.io/docs/cyclonedds/latest/installation/installation.html.

```bash
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake -DBUILD_EXAMPLES=ON -DCMAKE_INSTALL_PREFIX=$HOME/Documents/GitHub/cyclonedds/install ..
cmake --build . --target install
```

At this point, if you are connected to a robot and you run `./ddsperf sanity` (located in `$HOME/Documents/GitHub/cyclonedds/install/bin/` or whatever you chose) you should start to see data:

```
[8268] 2.005  rss:4.7MB vcsw:0 ivcsw:472 tev:0%+1% recv:0%+1%
[8268] 3.005  rss:4.8MB vcsw:0 ivcsw:275 tev:0%+1% others:0%+1%
```

#### List CycloneDDS Topics on Mac

```bash
export CYCLONEDDS_HOME=$HOME/Documents/GitHub/cyclonedds/install
export CMAKE_PREFIX_PATH=$HOME/Documents/GitHub/cyclonedds/install

# Then, set your wired Ethernet adapter to `192.168.123.99` and `255.255.255.0`, double check via `ifconfig`, and set the correct Ethernet adapter name in `NetworkInterface`:

export CYCLONEDDS_URI='
<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface name="en0" priority="default" multicast="default" />
      </Interfaces>
    </General>
    <Discovery>
      <EnableTopicDiscoveryEndpoints>true</EnableTopicDiscoveryEndpoints>
    </Discovery>
  </Domain>
</CycloneDDS>'
```

Then, compile and run the listtopics example:
```bash
cd $HOME/Documents/GitHub/cyclonedds/install/share/CycloneDDS/examples/listtopics
cmake .
cmake --build .
```

Now, when you run `./listtopics`, you should see an extensive data dump:

```bash
alive: ea9d27e0:9769a902:84fcfb59:2b6083bc rt/lf/lowstate unitree_go::msg::dds_::LowState_
alive: 3b07fe34:41cac3a8:9f12e6e6:f719133e rt/api/motion_switcher/response unitree_api::msg::dds_::Response_
alive: eec56342:f93120c6:bab05432:a1b4a0f8 rt/api/motion_switcher/request unitree_api::msg::dds_::Request_
alive: 8e9380cf:5e6f9214:9d5e768f:dfd6d595 rt/utlidar/voxel_map sensor_msgs::msg::dds_::PointCloud2_
alive: fee410f9:eaab3e20:b1c6a349:ac7b8b8c rt/utlidar/voxel_map_compressed unitree_go::msg::dds_::VoxelMapCompressed_
alive: d0709b33:bfcd6551:4d85a309:74df1cbb rt/utlidar/height_map sensor_msgs::msg::dds_::PointCloud2_
alive: 9e531810:a229c451:3315fd9c:415edc24 rt/utlidar/range_map sensor_msgs::msg::dds_::PointCloud2_
alive: 0a6d5257:787977ea:5ac0156c:1b39ba46 rt/utlidar/range_info geometry_msgs::msg::dds_::PointStamped_
alive: 83ebb9e5:6d69cbf8:4b458542:dae1be14 rt/utlidar/height_map_array unitree_go::msg::dds_::HeightMap_
alive: 1bf83b4b:562d97ef:9df5443a:84971a8a rt/utlidar/map_state unitree_go::msg::dds_::VoxelHeightMapState_
alive: a71b9ea9:984d116d:a2400d39:0839ce01 rt/utlidar/grid_map sensor_msgs::msg::dds_::PointCloud2_
alive: 3db4a0f1:1e75176b:ed20434a:fe87a63f rt/utlidar/robot_odom nav_msgs::msg::dds_::Odometry_
alive: 036f7221:ad729459:9435a047:09a9934e rt/utlidar/cloud_deskewed sensor_msgs::msg::dds_::PointCloud2_
alive: aa29f45f:8ce692b7:22cfb924:75a5bc71 rt/utlidar/mapping_cmd std_msgs::msg::dds_::String_
alive: 33264264:c316c9f9:525f6b90:cb1eb2ec rt/wirelesscontroller unitree_go::msg::dds_::WirelessController_
alive: 9613c077:46dfce42:88f8e974:9c7c8717 rt/api/sport/request unitree_api::msg::dds_::Request_
alive: d994fe76:2755380f:1d1835bb:ddba9c93 rt/api/obstacles_avoid/request unitree_api::msg::dds_::Request_
...
```

The Ethernet adapter you set above (such as `en0`) is the value you should provide to `/config/unitree_g1_humanoid.json`.

Then add the optional Python CycloneDDS module to OM1:

```bash
uv pip install -r pyproject.toml --extra dds
```

### Installation on Linux

You will need:

* OM1
* uv
* ffmpeg (for audio, otherwise the audio out will not work due to missing `ffprobe`)
* v4l-utils  (for video)
* CycloneDDS (for DDS comms to the G1 motion client)

```bash
sudo apt-get update
sudo apt-get install ffmpeg v4l-utils
```
v4l-utils is also useful to debug video problems. **WARNING**: The camera system, if not correctly configured, has a tendency to bring down the entire USB bus. FIX: reboot the humanoid.

Set the correct `CYCLONEDDS_HOME` env var. This is where the actual CycloneDDS is installed on your computer:
```bash
export CYCLONEDDS_HOME="$HOME/unitree_ros2/cyclonedds_ws/install/cyclonedds"
```
If you do not do this correctly, installation of the Python CycloneDDS, a later step, will fail since it cannot find the correct libraries. Note: make absolutely sure `CYCLONEDDS_HOME` actually points to the CycloneDDS install /lib. This can be confusing, since if you install indirectly via `unitree_ros2`, then the location of the CycloneDDS libraries will be in slightly different location than if you install directly, via `git clone https://github.com/eclipse-cyclonedds/cyclonedds`.

Then add the optional Python CycloneDDS module to OM1:

```bash
uv pip install -r pyproject.toml --extra dds
```

Note: on first invocation, the system sometimes cannot find the Unitree libraries. This should resolve by itself quickly.

## ORIN System Description

Your development computer will (should) be at `192.168.123.99`

The LIDAR is `192.168.123.120`

The internal control computer (RockChip, aka the `operation and control computing unit`) is at `192.168.123.161`

The internal development computer (Orin 16GB, aka the `development computing unit`) is at `192.168.123.164`

Useful commands:

```bash
sudo nmcli radio wifi on # Turn the wifi on/off
sudo nmcli device wifi connect XXXXX password XXXXX # Join WiFi network
sudo timedatectl set-ntp yes # Set time via NTP
```

### ORIN Set default input and output Audio devices

```bash
pactl list sources short                      # List input (microphone) devices
pactl list sinks short                        # List output (speaker) devices
pactl set-default-sink [SINK_NAME || SINK_ID] # Set default output device
pactl set-default-source [SOURCE_NAME || SOURCE_ID] # Set default input device
pactl set-sink-volume @DEFAULT_SINK@ 70%      # Set default output volume
```

## Control via Unitree Hand Controller

Hang G1 on gantry\
Turn on (short press, long press)\
Wait for boot to complete\
When the G1 boots, it is in `damp` state\
Use the hand controller to command "L1+A" and "L1+UP"\

The system is then ready to move using the `ai_sport` client. The system will respond to manual controller and SDK commands.

* Press "L1+A" -> EMERGENCY DAMP / SINK TO FLOOR
* Press "L1+UP" -> Stand firmly (aka "lock stand"). The arms will move slightly. The system is now in the "ready" state.
* Lower the G1 to the ground (but do not unclip her yet). Stability not yet running - she will fall over if let go.
* Press "R2+X" -> Start motion control. The arms will jump outwards and she will actively control her stability.
* Press "Start" to switch back and forth between `stand` and `step in place`.

Other actions:

* "SELECT + Y" -> Wave Hand. Alternates sides.
* "SELECT + A" -> Handshake. Hold for movement to complete. Wait 3s and press again to relax arm to initial state.
* "SELECT + X" -> Turn around and wave hand.s

Use the joysticks to move forwards and backwards, and to rotate/turn

## Boot from Chair

This is similar to boot from gantry, except, when you press "L1+UP", you have to help the humanoid stand up, while it straightens itself. For the sit-down procedure, back up the chair behind the robot, select "L1+LEFT", and help the humanoid settle back into the chair.

## Special DEBUG state

**Avoid this mode** since it disables all high level motion since it turns off the `ai_sport` client.

* Press L2+R2 -> Enter DEBUG STATE
* Press L2+A  -> Diagnostic Posture (Arms bent)
* Press L2+B  -> Relax arms, damping state

To exit this mode, reboot the G1.

## Using the Internal Orin

SSH to Orin via

```bash
ssh unitree@192.168.123.164
```

The default password is `123` but you should obviously change this. Result:

```
Welcome to Ubuntu 20.04.6 LTS (GNU/Linux 5.10.104-tegra aarch64)
Last login: Thu Jan  1 08:30:46 1970
ros:foxy(1) noetic(2) ?
```

Select `foxy(1)`.

## Fixing the broken CycloneDDS installation on the Nvidia Orin

The default installation of CycloneDDS on the G1 Orin is broken, since it does not support the newer `unitree_hg` IDL data format for the G1. Solution: remove the default CycloneDDS installation and reinstall following the [Unitree ROS2 installation instructions](https://github.com/unitreerobotics/unitree_ros2). The `unitree_hg` bug was fixed in early Dec. 2024 in this commit: unitreerobotics/unitree_ros2@b34fdf7.

You will need to export suitable env variables and correct the setting in `.bashrc` and in `$HOME/unitree_ros2/setup.sh`. Add this to the `.bashrc`:

```bash
export CYCLONEDDS_HOME="$HOME/unitree_ros2/cyclonedds_ws/install/cyclonedds"
```

Set `$HOME/unitree_ros2/setup.sh` to

```bash
#!/bin/bash
echo "Setup Unitree ROS2 Environment"
source /opt/ros/foxy/setup.bash
source $HOME/unitree_ros2/cyclonedds_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
<NetworkInterface name="eth0" priority="default" multicast="default" />
</Interfaces></General></Domain></CycloneDDS>'
```

Source the `setup.sh` via `source ~/unitree_ros2/setup.sh`. Finally, you should start to see data.

```bash ros2 topic echo lowstate
version:
- 0
- 0
mode_pr: 0
mode_machine: 4
tick: 1120016
imu_state:
  quaternion:
  - 0.9940093159675598
  - 0.0074933432042598724
  - -0.0023974007926881313
  - -0.10901317745447159
  gyroscope:
  - -0.00523598724976182
  - -0.00523598724976182
  - -0.0017453291220590472
  accelerometer:
  - 0.009999999776482582
  - -0.05000000074505806
  - 9.829999923706055
  rpy:
  - 0.015420285053551197
  - -0.003132336074486375
  - -0.21849146485328674
  temperature: 78
motor_state:
- mode: 1
  q: -0.008177042007446289
  dq: 0.0
  ddq: 0.0
  tau_est: -0.11186079680919647
  temperature:
  - 29
  - 28
  vol: 49.0
  sensor:
  - 0
  - 0
  motorstate: 0
  reserve:
  - 0
  - 1142
  - 4
  - 0
```

```bash ros2 topic echo lowcmd
mode_pr: 0
mode_machine: 4
motor_cmd:
- mode: 1
  q: 0.0
  dq: 0.0
  tau: 0.0
  kp: 0.0
  kd: 0.0
  reserve: 0
```

```bash ros2 topic echo odommodestate
stamp:
  sec: 0
  nanosec: 0
error_code: 0
imu_state:
  quaternion:
  - 0.9981062412261963
  - 0.007613412104547024
  - -0.0016289741033688188
  - -0.06102222576737404
  gyroscope:
  - 0.0
  - 0.0
  - 0.0
  accelerometer:
  - 0.0
  - 0.0
  - 0.0
  rpy:
  - 0.01539743971079588
  - -0.002322605811059475
  - -0.12214189022779465
  temperature: 0
mode: 0
progress: 0.0
gait_type: 0
foot_raise_height: 0.0
position:
- 0.0013194791972637177
- -0.018103253096342087
- 0.7247929573059082
body_height: 0.0
velocity:
- 6.116795248090057e-07
- -1.2412459682309418e-06
- -4.238392648403533e-05
yaw_speed: 0.0017453291220590472
range_obstacle:
- 0.0
...
foot_force:
- 0
- 0
- 0
- 0
foot_position_body:
- 0.0
...
foot_speed_body:
- 0.0
...
```

```bash ros2 topic list
/EstimatorData
/SymState
/SymState_back
/api/bashrunner/request
/api/bashrunner/response
/api/config/request
/api/config/response
/api/loco/request
/api/loco/response
/api/motion_switcher/request
/api/motion_switcher/response
/api/robot_state/request
/api/robot_state/response
/arm_sdk
/audiosender
/config_change_status
/dex3/left/cmd
/dex3/left/state
/dex3/right/cmd
/dex3/right/state
/frontvideostream
/gnss
/lf/bmsstate
/lf/dex3/left/state
/lf/dex3/right/state
/lf/lowstate
/lf/lowstate_doubleimu
/lf/mainboardstate
/lf/odommodestate
/loco_sdk
/lowcmd
/lowstate
/lowstate_doubleimu
/multiplestate
/odommodestate
/parameter_events
/public_network_status
/rosout
/rtc/state
/rtc_status
/selftest
/servicestate
/servicestateactivate
/videohub/inner
/webrtcreq
/webrtcres
/wirelesscontroller
```

Here is what will be visible on an external development machine at `.99`, for example, using `./bin/listtopics`:
```bash
rt/dex3/right/state unitree_hg::msg::dds_::HandState_
rt/lf/dex3/right/state unitree_hg::msg::dds_::HandState_
rt/dex3/right/cmd unitree_hg::msg::dds_::HandCmd_
rt/EstimatorData unitree_go::msg::dds_::EstimatorData_
rt/SymState_back unitree_go::msg::dds_::SymState_
rt/odommodestate unitree_go::msg::dds_::SportModeState_
rt/lf/odommodestate unitree_go::msg::dds_::SportModeState_
rt/lowstate unitree_hg::msg::dds_::LowState_
rt/SymState unitree_go::msg::dds_::SymState_
rt/api/bashrunner/response unitree_api::msg::dds_::Response_
rt/selftest std_msgs::msg::dds_::String_
rt/api/bashrunner/request unitree_api::msg::dds_::Request_
rt/lf/lowstate unitree_hg::msg::dds_::LowState_
rt/api/motion_switcher/response unitree_api::msg::dds_::Response_
rt/api/motion_switcher/request unitree_api::msg::dds_::Request_
rt/config_change_status unitree_go::msg::dds_::ConfigChangeStatus_
rt/api/config/response unitree_api::msg::dds_::Response_
rt/api/config/request unitree_api::msg::dds_::Request_
rt/api/robot_state/response unitree_api::msg::dds_::Response_
rt/api/robot_state/request unitree_api::msg::dds_::Request_
rt/servicestate std_msgs::msg::dds_::String_
rt/multiplestate std_msgs::msg::dds_::String_
rt/public_network_status std_msgs::msg::dds_::String_
rt/gnss std_msgs::msg::dds_::String_
rt/lf/bmsstate unitree_hg::msg::dds_::BmsState_
rt/lf/mainboardstate unitree_hg::msg::dds_::MainBoardState_
rt/webrtcreq std_msgs::msg::dds_::String_
rt/webrtcres std_msgs::msg::dds_::String_
rt/lowcmd unitree_hg::msg::dds_::LowCmd_
rt/lowstate_doubleimu unitree_hg_doubleimu::msg::dds_::doubleIMUState_
rt/lf/lowstate_doubleimu unitree_hg_doubleimu::msg::dds_::doubleIMUState_
rt/wirelesscontroller unitree_go::msg::dds_::WirelessController_
rt/frontvideostream unitree_go::msg::dds_::Go2FrontVideoData_
rt/audiosender unitree_go::msg::dds_::AudioData_
rt/servicestateactivate std_msgs::msg::dds_::String_
rt/rtc_status std_msgs::msg::dds_::String_
rt/videohub/inner std_msgs::msg::dds_::String_
rt/rtc/state std_msgs::msg::dds_::String_
rt/api/bashrunner/request unitree_api::msg::dds_::Request_
rt/lf/dex3/left/state unitree_hg::msg::dds_::HandState_
rt/dex3/left/state unitree_hg::msg::dds_::HandState_
rt/dex3/left/cmd unitree_hg::msg::dds_::HandCmd_
```

## Terminal based setup of Bluetooth audio devices

This is only needed on the headless Orin, otherwise (e.g. on the Mac) just use the system settings.
```bash
bluetoothctl

list    # show all paired devices
scan on # search for nearby devices
# once you have found the right device, you can then pair it
# many devices also require 'trusting' them

trust <MAC>
pair <MAC>
connect <MAC>
```
