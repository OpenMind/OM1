## go2_urdf
This repository contains the urdf model of go2.


## Build the library
Create a new catkin workspace:
```
# Create the directories
# Do not forget to change <...> parts
mkdir -p <directory_to_ws>/<catkin_ws_name>/src
cd <directory_to_ws>/<catkin_ws_name>/

# Initialize the catkin workspace
catkin init
```

Clone this library:
```
# Navigate to the directory of src
# Do not forget to change <...> parts
cd <directory_to_ws>/<catkin_ws_name>/src
git clone git@github.com:unitreerobotics/go2_urdf.git
```

Build:
```
# Build it
catkin build

# Source it
source <directory_to_ws>/<catkin_ws_name>/devel/setup.bash
```



## Run the library
```
# Show urdf model of go2 in Rviz
roslaunch go2_description go2_rviz.launch

```

## When used for isaac gym or other similar engine

Collision parameters in urdf can be amended to better train the robot:

Open "go2_description.urdf" in "./go2_description/urdf",
and amend the ` box size="0.213 0.0245 0.034" ` in links of "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh".

For example, change previous values to ` box size="0.11 0.0245 0.034" ` means the length of the thigh is shortened from 0.213 to 0.11, which can avoid unnecessary collision between the thigh link and the calf link.

The collision model before and after the above amendment are shown as "Normal_collision_model.png" and "Amended_collision_model.png" respectively.
