A more detailed unitree_guide document is available at unitree [developer site](https://support.unitree.com/home/zh/Algorithm_Practice).( only Chinese version now )

# Overview

The unitree_guide is an open source project for controlling the quadruped robot of Unitree Robotics, and it is also the software project accompanying [《四足机器人控制算法--建模、控制与实践》](https://detail.tmall.com/item.htm?spm=a212k0.12153887.0.0.5487687dBgiovR&id=704510718152) published by Unitree Robotics.

# Quick Start
The following will quickly introduce the use of unitree_guide in the gazebo simulator. For more usage, please refer to 《四足机器人控制算法--建模、控制与实践》.
## Environment
We recommend users to run this project in Ubuntu 18.04 and ROS melodic environment.
## Dependencies
1. [unitree_guide](https://github.com/unitreerobotics/unitree_guide)<br>
2. [unitree_ros](https://github.com/unitreerobotics/unitree_ros)<br>
3. [unitree_legged_msgs](https://github.com/unitreerobotics/unitree_ros_to_real)(Note that: unitree_legged_real package should not be a part of dependencies)<br>

Put these three packages in the src folder of a ROS workspace.

## build
Open a terminal and switch the directory to the ros workspace containing unitree_guide,  then run the following command to build the project:
```
catkin_make
```
If you have any error in this step, you can raise an issue to us.
## run
In the same terminal, run the following command step by step:
```
source ./devel/setup.bash
```
To open the gazebo simulator, run:
```
roslaunch unitree_guide gazeboSim.launch
```

For starting the controller, open another terminal and switch to the same directory,  then run the following command:
```
./devel/lib/unitree_guide/junior_ctrl
```

## usage
After starting the controller,  the robot will lie on the ground of the simulator, then press the '2' key on the keyboard to switch the robot's finite state machine (FSM) from **Passive**(initial state) to **FixedStand**,  then press the '4' key to switch the FSM from **FixedStand** to **Trotting**, now you can press the 'w' 'a' 's' 'd' key to control the translation of the robot, and press the 'j' 'l' key to control the rotation of the robot. Press the Spacebar, the robot will stop and stand on the ground
. (If there is no response, you need to click on the terminal opened for starting the controller and then repeat the previous operation)

# Note
Unitree_guide provides a basic quadruped robot controller for beginners. To achieve better performance, additional fine tuning of parameters or more advanced methods (such as MPC etc.) might be required. Any contribution and good idea from the robotics community are all welcome. Feel free to raise an issue ~ <br>
