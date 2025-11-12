# Use the official ROS Noetic full desktop image
FROM ros:noetic-ros-base-focal

# Set environment variables for ROS and DEBIAN frontend
ENV ROS_DISTRO=noetic \
    DEBIAN_FRONTEND=noninteractive

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    && rm -rf /var/lib/apt/lists/*

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

# Update and install required packages, including Gazebo 11.14.0
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    lsb-release \
    gnupg2 \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Install additional useful ROS packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-gazebo-ros-pkgs \
        ros-${ROS_DISTRO}-gazebo-ros-control \
        ros-${ROS_DISTRO}-joint-state-controller \
        ros-${ROS_DISTRO}-robot-state-publisher \
        ros-${ROS_DISTRO}-xacro \
        ros-${ROS_DISTRO}-move-base-msgs \
        liblcm-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Mesa utilities and software rendering libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libglx-mesa0 \
    libosmesa6 \
    libgles2-mesa \
    freeglut3 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# COPY ./start.sh /app/start.sh
COPY ./guide_ws /app/guide_ws

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && cd /app/guide_ws && catkin_make"

# Setup bashrc with ROS environment
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /app/guide_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "# Auto-detect GPU and set rendering mode" >> /root/.bashrc && \
    echo "if nvidia-smi &>/dev/null; then" >> /root/.bashrc && \
    echo "    echo 'NVIDIA GPU detected - using hardware acceleration'" >> /root/.bashrc && \
    echo "    export __GLX_VENDOR_LIBRARY_NAME=nvidia" >> /root/.bashrc && \
    echo "    export __NV_PRIME_RENDER_OFFLOAD=1" >> /root/.bashrc && \
    echo "    unset LIBGL_ALWAYS_SOFTWARE" >> /root/.bashrc && \
    echo "else" >> /root/.bashrc && \
    echo "    echo 'No GPU detected - using software rendering'" >> /root/.bashrc && \
    echo "    export LIBGL_ALWAYS_SOFTWARE=1" >> /root/.bashrc && \
    echo "    export MESA_GL_VERSION_OVERRIDE=3.3" >> /root/.bashrc && \
    echo "    export MESA_LOADER_DRIVER_OVERRIDE=swrast" >> /root/.bashrc && \
    echo "    export GALLIUM_DRIVER=llvmpipe" >> /root/.bashrc && \
    echo "    export __GLX_VENDOR_LIBRARY_NAME=mesa" >> /root/.bashrc && \
    echo "fi" >> /root/.bashrc && \
    echo "export QT_X11_NO_MITSHM=1" >> /root/.bashrc

COPY ./stream /workspace/stream

# Create startup script that auto-detects GPU
RUN echo '#!/bin/bash\n\
# Auto-detect GPU availability\n\
if [ -z "$FORCE_SOFTWARE_RENDERING" ]; then\n\
    if nvidia-smi &>/dev/null; then\n\
        echo "NVIDIA GPU detected - using hardware acceleration"\n\
        export __GLX_VENDOR_LIBRARY_NAME=nvidia\n\
        export __NV_PRIME_RENDER_OFFLOAD=1\n\
        export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}\n\
        export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all}\n\
        unset LIBGL_ALWAYS_SOFTWARE\n\
    else\n\
        echo "No NVIDIA GPU detected - using software rendering"\n\
        export LIBGL_ALWAYS_SOFTWARE=1\n\
        export MESA_GL_VERSION_OVERRIDE=3.3\n\
        export MESA_LOADER_DRIVER_OVERRIDE=swrast\n\
        export GALLIUM_DRIVER=llvmpipe\n\
        export __GLX_VENDOR_LIBRARY_NAME=mesa\n\
    fi\n\
else\n\
    echo "Forcing software rendering as requested"\n\
    export LIBGL_ALWAYS_SOFTWARE=1\n\
    export MESA_GL_VERSION_OVERRIDE=3.3\n\
    export MESA_LOADER_DRIVER_OVERRIDE=swrast\n\
    export GALLIUM_DRIVER=llvmpipe\n\
    export __GLX_VENDOR_LIBRARY_NAME=mesa\n\
fi\n\
\n\
# Always set Qt environment\n\
export QT_X11_NO_MITSHM=1\n\
\n\
# Execute the main command\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Set the default command
CMD ["/bin/bash"]
