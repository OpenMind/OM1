FROM ros:humble-ros-base-jammy
# The above base image is multi-platform (works on ARM64 and AMD64):
# Docker will automatically select the correct platform variant based on the host's architecture.

# Make sure bash catches errors (no need to chain commands with &&, use ; instead)
SHELL ["/bin/bash", "-o", "pipefail", "-o", "errexit", "-c"]


###########################
# 1.) Bring system up to the latest ROS desktop configuration
###########################
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install ros-humble-desktop

###########################
# 2.) Temporarily remove ROS2 apt repository
###########################
# The ROS2 repo is configured via a .sources file (DEB822 format)
RUN echo "Temporarily removing ROS2 apt repository..." && \
    if [ -L /etc/apt/sources.list.d/ros2.sources ]; then \
        rm /etc/apt/sources.list.d/ros2.sources && \
        echo "Removed ROS2 sources symlink"; \
    elif [ -f /etc/apt/sources.list.d/ros2.sources ]; then \
        mv /etc/apt/sources.list.d/ros2.sources /root/ros2.sources && \
        echo "Moved ROS2 sources file"; \
    fi

RUN apt-get update

###########################
# 3.) comment out the catkin conflict
###########################
RUN sed  -i -e 's|^Conflicts: catkin|#Conflicts: catkin|' /var/lib/dpkg/status
RUN apt-get install -f

###########################
# 4.) force install these packages
###########################
RUN apt-get download python3-catkin-pkg
RUN apt-get download python3-rospkg
RUN apt-get download python3-rosdistro
RUN dpkg --force-overwrite -i python3-catkin-pkg*.deb
RUN dpkg --force-overwrite -i python3-rospkg*.deb
RUN dpkg --force-overwrite -i python3-rosdistro*.deb
RUN apt-get install -f

###########################
# 5.) Install the latest ROS1 desktop configuration
# see https://packages.ubuntu.com/jammy/ros-desktop-dev
# note: ros-desktop-dev automatically includes tf tf2
###########################
RUN apt-get -y install ros-desktop-dev

# fix ARM64 pkgconfig path issue -- Fix provided by ambrosekwok
RUN if [[ $(uname -m) = "arm64" || $(uname -m) = "aarch64" ]]; then                     \
      cp /usr/lib/x86_64-linux-gnu/pkgconfig/* /usr/lib/aarch64-linux-gnu/pkgconfig/;   \
    fi

###########################
# 6.) Restore the ROS2 apt repos and set compilation options.
#   For example, to include ROS tutorial message types, pass
#   "--build-arg ADD_ros_tutorials=1" to the docker build command.
###########################
# Restore the ROS2 repository symlink
RUN echo "Restoring ROS2 apt repository..." && \
    if [ -f /usr/share/ros-apt-source/ros2.sources ]; then \
        ln -s /usr/share/ros-apt-source/ros2.sources /etc/apt/sources.list.d/ros2.sources && \
        echo "Restored ROS2 sources symlink"; \
    elif [ -f /root/ros2.sources ]; then \
        mv /root/ros2.sources /etc/apt/sources.list.d/ros2.sources && \
        echo "Restored ROS2 sources file"; \
    else \
        echo "Warning: Could not restore ROS2 sources"; \
    fi

RUN apt-get -y update


###########################
# 7.) Compile ros1_bridge
###########################
RUN                                                                             \
    #-------------------------------------                                      \
    # Apply the ROS2 underlay                                                   \
    #-------------------------------------                                      \
    source /opt/ros/humble/setup.bash;                                          \
    #-------------------------------------                                      \
    # Finally, build the Bridge                                                 \
    #-------------------------------------                                      \
    mkdir -p /ros-humble-ros1-bridge/src;                                       \
    cd /ros-humble-ros1-bridge/src;                                             \
    git clone https://github.com/smith-doug/ros1_bridge.git;                    \
    cd ros1_bridge/;                                                            \
    git checkout action_bridge_humble;                                          \
    cd ../..;                                                                   \
    MEMG=$(printf "%.0f" $(free -g | awk '/^Mem:/{print $2}'));                 \
    NPROC=$(nproc);  MIN=$((MEMG<NPROC ? MEMG : NPROC));                        \
    echo "Please wait...  running $MIN concurrent jobs to build ros1_bridge";   \
    time MAKEFLAGS="-j $MIN" colcon build --event-handlers console_direct+      \
      --cmake-args -DCMAKE_BUILD_TYPE=Release

###########################
# 8.) Clean up
###########################
RUN apt-get -y clean all; apt-get -y update

###########################
# 9.) Pack all ROS1 dependent libraries
###########################
# fix ARM64 pkgconfig path issue -- Fix provided by ambrosekwok
RUN if [[ $(uname -m) = "arm64" || $(uname -m) = "aarch64" ]]; then                    \
      cp /usr/lib/x86_64-linux-gnu/pkgconfig/* /usr/lib/aarch64-linux-gnu/pkgconfig/;  \
    fi

RUN ROS1_LIBS="libxmlrpcpp.so";                                                 \
     ROS1_LIBS="$ROS1_LIBS librostime.so";                                      \
     ROS1_LIBS="$ROS1_LIBS libroscpp.so";                                       \
     ROS1_LIBS="$ROS1_LIBS libroscpp_serialization.so";                         \
     ROS1_LIBS="$ROS1_LIBS librosconsole.so";                                   \
     ROS1_LIBS="$ROS1_LIBS librosconsole_log4cxx.so";                           \
     ROS1_LIBS="$ROS1_LIBS librosconsole_backend_interface.so";                 \
     ROS1_LIBS="$ROS1_LIBS liblog4cxx.so";                                      \
     ROS1_LIBS="$ROS1_LIBS libcpp_common.so";                                   \
     ROS1_LIBS="$ROS1_LIBS libb64.so";                                          \
     ROS1_LIBS="$ROS1_LIBS libaprutil-1.so";                                    \
     ROS1_LIBS="$ROS1_LIBS libapr-1.so";                                        \
     ROS1_LIBS="$ROS1_LIBS libactionlib.so.1d";                                 \
     cd /ros-humble-ros1-bridge/install/ros1_bridge/lib;                        \
     for soFile in $ROS1_LIBS; do                                               \
       soFilePath=$(ldd libros1_bridge.so | grep $soFile | awk '{print $3;}');  \
       cp $soFilePath ./;                                                       \
     done

###########################
# 10.) Install additional dependencies
###########################
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    ros-humble-rosidl-generator-dds-idl

COPY ./messages /app/messages

# do a clean rebuild of the bridge, this is to be optimised.
RUN cd /app/messages/ros1 &&\
    unset ROS_DISTRO &&\
    time colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release &&\
    cd /app/messages/ros2 &&\
    source /opt/ros/humble/setup.bash &&\
    time colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release &&\
    cd /ros-humble-ros1-bridge &&\
    rm -rf build install log &&\
    source /app/messages/ros1/install/setup.bash &&\
    source /app/messages/ros2/install/setup.bash &&\
    time colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
# ignore the ACTION_GOAL_RESPONSE_CB_CONST error

COPY ./recast /app/recast

RUN cd /app/recast &&\
    source /opt/ros/humble/setup.bash &&\
    colcon build

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros-humble-ros1-bridge/install/setup.bash" >> /root/.bashrc && \
    echo "source /app/recast/install/setup.bash" >> /root/.bashrc

# Set the default command
CMD ["/bin/bash"]
