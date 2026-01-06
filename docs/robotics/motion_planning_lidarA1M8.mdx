---
title: RPLidar A1M8
description: "RPLidar A1M8 Setup and Use"
---

## RPLidar

These instructions are for an SLAMTEC RPLIDAR A1M8. This unit is used on the TurtleBot4 and can be added to the Unitree Go2.

## Architecture

The collision avoidance and path checking code pre-computes 9 different paths, 4 to the left, one straight ahead, 4 to the right, and one to the back. For each of the 9 possible paths, the code checks whether the path approaches any detected object to within `half_width_robot`. If not, the path is considered to be a valid choice and the motion system can execute that path.

## Assumptions

The code assumes that any unpredictable barriers (e.g. humans crossing the path of the robot) will be avoided using separate code within the `action` driver, such as by issuing a "STOP" command when an object is detected in front of the robot.

## Setup

Run `rptest.py` (located in `system_hw_test`) to determine the `sensor_mounting_angle` and the `angles_blanked`. These values depend on how you mounted the sensor and the radial position of any fixed obstructions, such as mounting brackets.

```py
"""
Robot and sensor configuration
"""
half_width_robot = 0.20  # the width of the robot is 40 cm
relevant_distance_max = 1.1  # meters
relevant_distance_min = 0.16 # meters
sensor_mounting_angle = 180.0  # corrects for how sensor is mounted
angles_blanked = [[-180.0, -160.0], [32.0, 46.6]]
```

### How to determine the sensor_mounting_angle and the angles_blanked

- `sensor_mounting_angle`: The `sensor_mounting_angle` refers to the angle between the LIDAR sensor's scanning plane and the horizontal plane of your robot or platform. The `sensor_mounting_angle` can be read physically from the LIDAR. Refer to the following image for guidance:

![sensor_mounting_angle](../assets/lidar_angle.png)

- `angles_blanked`: The `angles_blanked` array can be used to prevent fixed obstructions in the field of view of the LIDAR from producing erroneous object avoidance messages.

## Unitree RPLidar

Determine the serial port the sensor is using:

```bash
ls /dev/tty.*
ls /dev/cu.*
```

Then, run

```bash
uv run rptest.py --serial /dev/cu.usbserial-0001
```

## TurtleBot4

By default, the TurtleBot4 is configured to use the RPLidar A1M8. You can verify this by running:

```bash
ls -l /dev/RPLIDAR
```

To see the raw LIDAR data, provide the robot's URID (such as `OM742d35Cc6634`) and run:

```bash
uv run rptest.py --URID OM123435Cc1234
```

Typically, this command would be executed on your laptop, connected to the TurtleBot4 via Zenoh.

## Using the RPLidar A1M8 in OM1

Configure the `.json5` as needed:

```python
"""
Robot and sensor configuration
"""
{
  "type": "RPLidar",
  "config": {
    "use_zenoh": true, # or false, if you are using serial to connect directly to the LIDAR
    "serial_port": "/dev/cu.usbserial-0001",
    "half_width_robot": 0.21, # the width of the robot is 40 cm
    "relevant_distance_max": 1.1, # meters
    "sensor_mounting_angle": 180.0, # corrects for how sensor is mounted relative to robot
    "angles_blanked": [[-180.0, -160.0], [32.0, 46.6]]
  }
}
```
