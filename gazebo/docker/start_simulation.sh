#!/bin/bash

# Stop existing containers if any
safe_docker_stop_and_remove() {
    local container_name=$1
    # Check if the container exists and is running, then stop it
    if docker ps -q -f name="$container_name" &>/dev/null; then
        echo "Stopping container: $container_name"
        docker stop "$container_name" &>/dev/null
    fi

    # Check if the container exists (stopped or running), then remove it
    if docker container ls -a -q -f name="$container_name" &>/dev/null; then
        echo "Removing container: $container_name"
        docker container rm "$container_name" &>/dev/null
    fi
}

# Stop and remove specific containers
safe_docker_stop_and_remove ros1_gazebo_container
safe_docker_stop_and_remove ros1_ros2_bridge

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the combined docker-compose file with an absolute path
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

# Auto-detect GPU and set Docker runtime AND rendering variables
if nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU detected - using nvidia runtime"
    export DOCKER_RUNTIME=nvidia
    # These will be passed to the container
    export USE_GPU_RENDERING=true
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
    export __NV_PRIME_RENDER_OFFLOAD=1
else
    echo "No NVIDIA GPU detected - using default runtime"
    export DOCKER_RUNTIME=runc
    export USE_GPU_RENDERING=false
fi

# Allow X server connections
xhost +local:docker 2>/dev/null || true

# Ensure containers are started (or restarted if already running)
echo "Ensuring containers are running..."
docker compose -f $COMPOSE_FILE up -d --build

# Wait for container to be ready
sleep 2

# Set GPU environment inside the container after it starts
if [ "$USE_GPU_RENDERING" = "true" ]; then
    echo "Configuring container for GPU rendering..."
    docker exec ros1_gazebo_container bash -c "
        echo 'export __GLX_VENDOR_LIBRARY_NAME=nvidia' >> /tmp/gpu_env
        echo 'export __NV_PRIME_RENDER_OFFLOAD=1' >> /tmp/gpu_env
        echo 'unset LIBGL_ALWAYS_SOFTWARE' >> /tmp/gpu_env
        source /tmp/gpu_env
    "
    echo "GPU rendering configured. You can verify with: docker exec ros1_gazebo_container glxinfo | grep 'OpenGL renderer'"
else
    echo "Software rendering will be used"
fi

echo "Containers are ready!"

# Wait for a moment to ensure containers are up
sleep 5

# # Close all gnome-terminal instances
# echo "Closing all gnome-terminal instances..."
# pkill -f gnome-terminal

# Wait a moment to ensure all terminals are closed
sleep 2

# Preparing

echo "Disabling access control"
xhost +

# Assume device number is 10
DEVICE_NO=10
echo "Creating virtual video device at /dev/video$DEVICE_NO"
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=$DEVICE_NO exclusive_caps=1

# Launching containers

echo "Launching Gazebo"
docker exec -d ros1_gazebo_container bash -c '\
source /opt/ros/noetic/setup.bash && \
source /app/guide_ws/devel/setup.bash && \
roslaunch unitree_guide gazeboSim.launch gui:=true'
sleep 12

echo "Launching control"
docker exec -d ros1_gazebo_container bash -c '\
source /opt/ros/noetic/setup.bash && \
source /app/guide_ws/devel/setup.bash && \
/app/guide_ws/devel/lib/unitree_guide/junior_ctrl;'
sleep 3

echo "Launching command publisher"
docker exec -d ros1_gazebo_container bash -c '\
source /opt/ros/noetic/setup.bash && \
source /app/guide_ws/devel/setup.bash && \
rosrun unitree_guide om1_publisher.py;'
sleep 1

echo "Launching camera streamer"
docker exec -d ros1_gazebo_container bash -c '\
source /opt/ros/noetic/setup.bash && \
python3 /workspace/stream/stream_camera.py;'
sleep 1

echo "Attaching to ros1_ros2_bridge..."
echo "Launching ros1 bridge"
docker exec -d ros1_ros2_bridge bash -c '\
source /opt/ros/humble/setup.bash && \
source /ros-humble-ros1-bridge/install/setup.bash && \
source /app/recast/install/setup.bash && \
ros2 run ros1_bridge dynamic_bridge --bridge-all-topics'

echo "Launching topic bridge"
docker exec -d ros1_ros2_bridge bash -c '\
source /opt/ros/humble/setup.bash && \
source /ros-humble-ros1-bridge/install/setup.bash && \
source /app/recast/install/setup.bash && \
ros2 run om1_topic_bridge om1_topic_bridge'

# Notify the user
echo "Simulation is running."

echo "Capturing video"
ffmpeg -re -f mjpeg -i tcp://127.0.0.1:5000 -pix_fmt yuyv422 -f v4l2 /dev/video$DEVICE_NO > /dev/null 2>&1 &

echo "Launching move publisher"

docker exec -it ros1_gazebo_container bash -c '\
source /opt/ros/noetic/setup.bash && \
source /app/guide_ws/devel/setup.bash && \
rosrun move_publisher move_publisher.py; \
exec bash'
