---
title: Unitree Go2 Quadruped
description: "Unitree Go2 EDU Quadruped (dog)"
---

OM1 can control a Unitree Go2 EDU out of the box. This has been tested on Nvidia Orin, Mac Mini, and current (silicon) Mac laptops.

## Step 1 - Establishing Ethernet and DDS Connectivity

Connect the Unitree Go2 EDU to your development machine with an Ethernet cable. Open the network settings and find the network interface that is connected to the Go2 EDU. In the IPv4 settings, change the IPv4 mode to `manual`, set the address to `192.168.123.99`, and set the mask to `255.255.255.0`. After completion, click `apply` (or equivalent) and wait for the network to reconnect. Provide the name of the network adapter in the `"unitree_ethernet": "en0"` entry in the `unitree_go2.config` file.

Then, install [`CycloneDDS`](https://index.ros.org/p/cyclonedds/). `CycloneDDS` works on Mac, Linux, and PC. Run:

```bash
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=ON
cmake --build . --target install
```

Next, set `CYCLONEDDS_HOME`, `CMAKE_PREFIX_PATH`, and `CYCLONEDDS_URI` to the correct values for your computer. Example settings for a typical Mac installation are provided below. You should add these paths to your environment via your `.zshrc` or equivalent.

```bash
export CYCLONEDDS_HOME=$HOME/Documents/GitHub/cyclonedds/install

export CMAKE_PREFIX_PATH=$HOME/Documents/GitHub/cyclonedds/install

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

Then, compile and run the `listtopics` example:

```bash
cd $HOME/Documents/GitHub/cyclonedds/install/share/CycloneDDS/examples/listtopics
cmake .
cmake --build .
./listtopics
```

On Mac, you **might** need to `allow incoming connections` in the popup the first time you run `listtopics`.

Running `listtopics` should result in an extensive data dump of available topics:

```bash
alive: df4efe0e:812a86b1:647728be:c1f7a312 rt/lf/lowstate unitree_go::msg::dds_::LowState_
alive: c3a612c2:99b329c6:22510d3d:d385e1e6 rt/api/motion_switcher/response unitree_api::msg::dds_::Response_
alive: 78b1db7f:622f2dfe:fb4d9e8f:987083f0 rt/api/motion_switcher/request unitree_api::msg::dds_::Request_
alive: 84978827:0d460527:b4f054ac:e546468a rt/api/gpt/request unitree_api::msg::dds_::Request_
alive: 23d08ca2:bea974c8:d44a51c0:d2c3cf27 rt/api/gpt/response unitree_api::msg::dds_::Response_
alive: ac63fb3a:bb2b8a5b:c1a42ed4:bf470e91 rt/gptflowfeedback std_msgs::msg::dds_::String_
alive: fc5e351e:00f676bb:236ec7d8:da9f115b rt/api/sport/request unitree_api::msg::dds_::Request_
alive: b1be3e06:596bd40d:8ef51579:50245496 rt/api/sport/response unitree_api::msg::dds_::Response_
alive: da491277:72915ef6:c7407d32:fb2ce5fa rt/api/videohub/request unitree_api::msg::dds_::Request_
alive: 9072826b:ffcd904a:ba8a0092:792335d7 rt/api/videohub/response unitree_api::msg::dds_::Response_
alive: 077c7627:183247d6:2a698bc3:f8df4577 rt/utlidar/range_info geometry_msgs::msg::dds_::PointStamped_
alive: 188652eb:bf16b2a2:dbc4ac3a:af71d576 rt/lf/sportmodestate unitree_go::msg::dds_::SportModeState_
alive: 23dd31a3:b6828a1e:cb390d4f:1a9a1d87 rt/gpt_cmd std_msgs::msg::dds_::String_
alive: c53b329d:7aa96ef0:80c9d35d:d8459b42 rt/api/vui/request unitree_api::msg::dds_::Request_
alive: 87b1b32e:d9ec09ce:b928db03:eaf5f28f rt/api/vui/response unitree_api::msg::dds_::Response_
alive: 4d0e307c:c20328f4:0dd1efc5:7ca1e56c rt/mf/sportmodestate unitree_go::msg::dds_::SportModeState_
alive: 083d7026:d0ac857d:ad12c938:2b98f89a rt/utlidar/height_map_array unitree_go::msg::dds_::HeightMap_
alive: 06f05795:ebf8f837:fb736321:16cd723d rt/wirelesscontroller unitree_go::msg::dds_::WirelessController_
alive: 3f2bd6b9:22da59bf:ca7b2ec6:2ab25639 rt/api/obstacles_avoid/response unitree_api::msg::dds_::Response_
alive: 3392a31a:eec44934:a234ee4e:e9b0645d rt/api/obstacles_avoid/request unitree_api::msg::dds_::Request_
alive: 9c38174a:b9719d10:5ceaf5d6:58cad182 rt/api/config/request unitree_api::msg::dds_::Request_
alive: 243d1a9a:beb73e99:a7438b89:8881ba7a rt/api/config/response unitree_api::msg::dds_::Response_
alive: 1ffd3f85:9e4cbd10:b07286be:95b31d4c rt/api/sport_lease/response unitree_api::msg::dds_::Response_
alive: ab8cdb14:cb59d2c1:7dba035c:424e94c5 rt/api/sport_lease/request unitree_api::msg::dds_::Request_
alive: 232c059d:f9650b34:ee28aae8:da3ab7d6 rt/lowcmd unitree_go::msg::dds_::LowCmd_
alive: a9cf187b:009f8f56:bfdf14b4:c2525358 rt/sportmodestate unitree_go::msg::dds_::SportModeState_
alive: 282f8688:c4486d0f:ae222a71:f33dc3b9 rt/lowstate unitree_go::msg::dds_::LowState_
alive: e553c276:3cdb6d66:52898f8b:59b25a8c rt/config_change_status unitree_go::msg::dds_::ConfigChangeStatus_
alive: 699e3ec7:13005f67:4165667f:a84d4593 rt/webrtcreq std_msgs::msg::dds_::String_
alive: 921b611e:2649a718:a59ee5cd:22dc2309 rt/webrtcres std_msgs::msg::dds_::String_
alive: 4e0a5acf:15405a12:f6e929ea:5c320a0a rt/api/audiohub/request unitree_api::msg::dds_::Request_
alive: 873309ee:12ef5d7c:8cd557b7:00b1931c rt/api/audiohub/response unitree_api::msg::dds_::Response_
alive: e7a31796:64c222e8:a8f3e215:22e11e24 rt/rtc/state std_msgs::msg::dds_::String_
alive: 80ba3475:34843e73:c939481c:0db3198d rt/audiohub/player/state std_msgs::msg::dds_::String_
alive: 82fb9789:0b61fd9c:5a20c796:876944da rt/api/fourg_agent/response unitree_api::msg::dds_::Response_
alive: 63540cad:26e63d68:0b2a3b41:0a431b21 rt/api/fourg_agent/request unitree_api::msg::dds_::Request_
alive: 0582765e:bebc0d3c:31f2aa76:849d52d7 rt/public_network_status std_msgs::msg::dds_::String_
alive: 5a1e209f:77622587:dff56d06:100b934a rt/gnss std_msgs::msg::dds_::String_
alive: 5c4904ed:4cdcda1f:17498349:91cc9154 rt/api/uwbswitch/request unitree_api::msg::dds_::Request_
alive: 40ce3a5c:80f4f2e4:50679aee:073b34d7 rt/api/uwbswitch/response unitree_api::msg::dds_::Response_
alive: c2bd3891:436200a6:bcf7d9fe:67115b31 rt/api/bashrunner/response unitree_api::msg::dds_::Response_
alive: c0acff0c:48bc2d73:b9c0bc78:9a39ac86 rt/selftest std_msgs::msg::dds_::String_
alive: 22066ccf:337367fe:bfc8f45d:07cb2de6 rt/api/bashrunner/request unitree_api::msg::dds_::Request_
alive: e1235a04:2b4036af:97482629:7591bc73 rt/utlidar/cloud sensor_msgs::msg::dds_::PointCloud2_
alive: 7daf47c7:581b2844:c890ed12:95e30a2e rt/utlidar/cloud_deskewed sensor_msgs::msg::dds_::PointCloud2_
alive: 4a8d54dc:e5ee8a08:fe3219d9:07f76dcf rt/utlidar/lidar_state unitree_go::msg::dds_::LidarState_
alive: 44260812:9b9f6a04:af980abc:e73a77fe rt/utlidar/switch std_msgs::msg::dds_::String_
alive: 749c1f9a:4b0cec32:19b92633:17e62e1d rt/utlidar/robot_odom nav_msgs::msg::dds_::Odometry_
alive: 79ce8c20:07aae9f8:c523d89c:b110b187 rt/utlidar/robot_pose geometry_msgs::msg::dds_::PoseStamped_
alive: 2c16a46d:da94a86f:d52ef5cc:7051a9fa rt/utlidar/foot_position sensor_msgs::msg::dds_::PointCloud2_
alive: 4384ffd2:e20ddb0e:f7b76a2c:d2c6a30c rt/utlidar/imu sensor_msgs::msg::dds_::Imu_
alive: f3aab9a3:95247169:cfbf8263:c5a9b106 rt/utlidar/mapping_cmd std_msgs::msg::dds_::String_
alive: 22359611:515cc56c:deeb92c9:c94220cf rt/uslam/client_command std_msgs::msg::dds_::String_
alive: 3dcca3f4:78429e39:564ca64f:05a1ece3 rt/uslam/cloud_map sensor_msgs::msg::dds_::PointCloud2_
alive: 4cc72605:f2246ae7:b747729e:64efbe90 rt/uslam/server_log std_msgs::msg::dds_::String_
alive: ebdc8a4a:9e5c19b5:26543bcc:0fa10748 rt/utlidar/voxel_map sensor_msgs::msg::dds_::PointCloud2_
alive: d447621e:3988c1dc:f66702de:0686316c rt/utlidar/voxel_map_compressed unitree_go::msg::dds_::VoxelMapCompressed_
alive: 5e812cab:094b8728:1b5e3e01:e016ee20 rt/utlidar/height_map sensor_msgs::msg::dds_::PointCloud2_
alive: 2de95d6a:9bb1ad47:ffd2e68a:e630fc84 rt/utlidar/range_map sensor_msgs::msg::dds_::PointCloud2_
alive: 46a6be60:38d89c4b:2c15e85e:1de4d940 rt/utlidar/map_state unitree_go::msg::dds_::VoxelHeightMapState_
alive: 35ccc911:8434f2c6:dcde737d:9e2be105 rt/utlidar/grid_map sensor_msgs::msg::dds_::PointCloud2_
alive: 1fdd190f:1809f14a:cbcf54a8:83382be7 rt/uwbstate unitree_go::msg::dds_::UwbState_
alive: 7474c70d:75a1ec58:39efc2e5:9dde2533 rt/api/programming_actuator/response unitree_api::msg::dds_::Response_
alive: 42699445:5dcf6a55:4ed227af:6ba4ae91 rt/api/programming_actuator/request unitree_api::msg::dds_::Request_
alive: 47a98910:be44d5c4:55ef536f:30e24971 rt/wireless_controller unitree_go::msg::dds_::WirelessController_
alive: 1c3b1069:4aa20466:30a3036b:c6b59fa8 rt/frontvideostream unitree_go::msg::dds_::Go2FrontVideoData_
alive: b998bf1b:6930232f:a9a62e3e:b1785122 rt/videohub/inner std_msgs::msg::dds_::String_
alive: 42c7a686:b0e92d30:fe95d5c0:9af26c6f rt/api/robot_state/response unitree_api::msg::dds_::Response_
alive: 09866cdd:dd97fa27:978c004f:3624dace rt/api/robot_state/request unitree_api::msg::dds_::Request_
alive: fb0e8c0f:5f108434:e87e9cf7:7848477d rt/servicestate std_msgs::msg::dds_::String_
alive: 6e9b68c3:8cb0df52:98603f8c:b934b1f6 rt/multiplestate std_msgs::msg::dds_::String_
alive: 6f564b14:8d2a936a:e3ac501c:d0fa740f rt/audiosender unitree_go::msg::dds_::AudioData_
alive: ab2e56b3:0906d3af:9a0efbb8:2c0fbd2b rt/uwbswitch unitree_go::msg::dds_::UwbSwitch_
alive: edf34ede:ec71ed87:ff6d918b:fd6177f9 rt/servicestateactivate std_msgs::msg::dds_::String_
alive: da41bb3b:e14530da:3288e103:dcf8888a rt/wirelesscontroller_unprocessed unitree_go::msg::dds_::WirelessController_
alive: 8a575fb2:ad676bfd:a3b20683:ae69b9af rt/rtc_status std_msgs::msg::dds_::String_
alive: 05d38f9b:d5113e6e:14601dbc:42e800ca rt/api/bashrunner/request unitree_api::msg::dds_::Request_
alive: 82a94751:dd399ada:7d9e6a9f:d7a0176e rt/api/gas_sensor/request unitree_api::msg::dds_::Request_
alive: ba193460:48bd7d2f:116da37d:f30b6c42 rt/qt_command unitree_interfaces::msg::dds_::QtCommand_
alive: d4f5d6cb:b6399171:13f983fb:45b46d7a rt/qt_add_node unitree_interfaces::msg::dds_::QtNode_
alive: 6f31b8ea:188ccefd:0bfd96dc:3a37a02f rt/qt_add_edge unitree_interfaces::msg::dds_::QtEdge_
alive: d1df34ff:1eab5cca:dd6efa75:2d6822cb rt/arm_Command unitree_arm::msg::dds_::ArmString_
alive: ea56b746:45db6798:5bc61259:56a9ced8 rt/programming_actuator/command std_msgs::msg::dds_::String_
alive: bcfd7290:715c3062:f749cd88:65a00f50 rt/api/gas_sensor/response unitree_api::msg::dds_::Response_
alive: cbda9e6d:70ba0985:dde9d637:788a68c1 rt/query_result_node unitree_interfaces::msg::dds_::QtNode_
alive: 3e9b2d75:a0c37b00:84a6a836:0d551674 rt/query_result_edge unitree_interfaces::msg::dds_::QtEdge_
alive: caa2c5ce:1b35a561:838107c5:426dc33b rt/qt_notice std_msgs::msg::dds_::String_
alive: d04fbb81:2460ecfd:65992300:0488c9e0 rt/pctoimage_local unitree_interfaces::msg::dds_::PcToImage_
alive: 0a0535a3:698337d4:c18b0e88:881cfe6e rt/lio_sam_ros2/mapping/odometry nav_msgs::msg::dds_::Odometry_
alive: 38c5e268:9cf0ebc1:740c30c6:aeb63f0a rt/arm_Feedback unitree_arm::msg::dds_::ArmString_
alive: ed0192ea:695ebbbd:9c0704a9:44a56576 rt/gas_sensor std_msgs::msg::dds_::String_
alive: 56b92f32:a4984539:7d02a691:72b8ed71 rt/uslam/frontend/cloud_world_ds sensor_msgs::msg::dds_::PointCloud2_
alive: 11641fd3:41e4e795:eb3a132d:6ba0b093 rt/uslam/frontend/odom nav_msgs::msg::dds_::Odometry_
alive: c0263c37:58c215e0:a5cf9018:0255fb8e rt/uslam/localization/odom nav_msgs::msg::dds_::Odometry_
alive: a3bb4e5c:ca3c645a:58b4a065:a84059ab rt/uslam/navigation/global_path sensor_msgs::msg::dds_::PointCloud2_
alive: 1453e55f:a4d5babe:aec670aa:6c0a9bff rt/uslam/localization/cloud_world sensor_msgs::msg::dds_::PointCloud2_
alive: 1a6535ff:d7346fd8:87885024:9a136972 rt/programming_actuator/feedback std_msgs::msg::dds_::String_
```

## Step 2 - Add the Python CycloneDDS module

Add the `dds` python module to your code base: `uv pip install -r pyproject.toml --extra dds`.

## Step 3 - Connect an Game Controller

On Mac, if you are using an Xbox controller, **make sure that it is running the most recent firmware**, otherwise you will be able to pair but not connect to the controller. You will need a PC with `Xbox Accessories` to update the Xbox controller's firmware. Many Mac OS updates hose support for the Xbox controller, which is then fixed (after some delay) via an Xbox firmware update.

On Mac, you will need to install `hidapi`:

```bash
brew install hidapi
```

NOTE: There is a bug on Mac when installing packages with `brew` - some libraries cannot be found by `uv`. If you get errors such as `Unable to load any of the following libraries:libhidapi-hidraw.so`, set `export DYLD_FALLBACK_LIBRARY_PATH=$HOMEBREW_PREFIX/lib` in your `.zshenv` or equivalent.

On Linux, install `hidapi` like this:

```bash
sudo apt-get update
sudo apt-get install python-dev libusb-1.0-0-dev libudev-dev libhidapi-dev
```

## Accessing Unitree Data

For debugging, you can access Unitree data as follows:

Front video feed:
```bash
uv run system_hw_test/go2_camera_opencv.py en0
```

Front video single image:
```bash
uv run system_hw_test/go2_capture_image.py en0
```

Lowstate system/joint data:
```bash
uv run system_hw_test/go2_data_stream.py en0
```

Lidar:
```bash
uv run system_hw_test/go2_lidar.py en0
```

Note - the internal Go2 LIDAR is currently not used - a separate RPLIDAR mounted to the top of the dog's head is used instead.

## Minimal Quadruped Functionality

In this configuration, the quadruped observes its environment, listens and speaks, but there is no AI-controlled movement. You can manually control the dog's movements with an game controller.

Run
```bash
uv run src/run.py unitree_go2_basic
```

Press:

* A to stand up
* B to sit down
* The D-pad allows you to steer the quadruped.

See the [Quadruped Configurations](unitree_go2_quadruped_configurations.mdx) for additional configurations.

## Unitree Go2 EDU Common Problems

*Channel factory init error*: If you see a `channel factory init error`, then you have not set the correct network interface adapter - the one you want to use is the network interface adapter *on your development machine - the computer you are currently sitting in front of* that is plugged into the Unitree quadruped (which has its own internal RockChip computer and network interface, which is *not* relevant to you right now). The Ethernet adapter - such as `eno0` or `en0` - needs to be set in the `"unitree_ethernet": "en0"` entry in the `unitree_go2.config` file.

*The CycloneDDS library could not be located*: You did not install CycloneDDS (see above), or, you did not provide a path to the `/install`, via `export CYCLONEDDS_HOME=$HOME/Documents/GitHub/cyclonedds/install` or equivalent.

*"nothing is working"* There are dozens of potential reasons "nothing is working". The first step is to test your ability to `ping` the quadruped motion control computer:

```bash ping
ping 192.168.123.161
```

Assuming you can `ping` the robot, then test the CycloneDDS middleware (see **STEP 1**). Once you see data flowing, then the rest of the system should work.
