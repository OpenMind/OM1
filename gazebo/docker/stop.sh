#!/bin/bash

docker stop ros1_gazebo_container
docker stop ros1_ros2_bridge
docker container rm ros1_gazebo_container
docker container rm ros1_ros2_bridge
