FROM ros:humble-ros-core AS base

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    libv4l-dev \
    python3-pip \
    build-essential \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y curl pkg-config libssl-dev \
&& curl https://sh.rustup.rs -sSf | sh -s -- -y \
&& . "$HOME/.cargo/env" \
&& echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> /root/.bashrc
ENV PATH="/root/.cargo/bin:${PATH}"

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10

RUN python3 -m pip install --upgrade pip

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
RUN git clone --branch releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds
WORKDIR /app/cyclonedds/build
RUN cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=ON \
 && cmake --build . --target install

ENV CYCLONEDDS_HOME=/app/cyclonedds/install \
    CMAKE_PREFIX_PATH=/app/cyclonedds/install \
    PYTHONPATH=/app/OM1:${PYTHONPATH} \
    ROS_DOMAIN_ID=0

WORKDIR /app/OM1
COPY . .

RUN git submodule update --init --recursive

RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'set -e' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo 'export CYCLONEDDS_HOME=/app/cyclonedds/install' >> /entrypoint.sh && \
    echo 'export CMAKE_PREFIX_PATH=/app/cyclonedds/install' >> /entrypoint.sh && \
    echo 'export PYTHONPATH=/app/OM1:${PYTHONPATH}' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo 'uv venv --clear' >> /entrypoint.sh && \
    echo 'uv pip install -r pyproject.toml --extra dds' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo 'exec uv run src/run.py "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["unitree_g1_humanoid"]