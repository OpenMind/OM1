FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    libv4l-dev \
    build-essential \
    cmake \
    python3-dev \
    libasound2 \
    libasound2-data \
    libasound2-plugins \
    libpulse0 \
    alsa-utils \
    alsa-topology-conf \
    alsa-ucm-conf \
    pulseaudio-utils \
    iputils-ping \
    curl \
    pkg-config \
    libssl-dev \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install uv (Python package & project manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ALSA / PulseAudio configuration
RUN mkdir -p /etc/alsa && ln -snf /usr/share/alsa/alsa.conf.d /etc/alsa/conf.d
RUN printf '%s\n' \
  'pcm.!default { type pulse }' \
  'ctl.!default { type pulse }' \
  > /etc/asound.conf

# Build CycloneDDS from source
WORKDIR /app
RUN git clone --branch releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds && \
    mkdir -p /app/cyclonedds/build && \
    cd /app/cyclonedds/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=ON && \
    cmake --build . --target install

# DDS env
ENV CYCLONEDDS_HOME=/app/cyclonedds/install \
    CMAKE_PREFIX_PATH=/app/cyclonedds/install \
    LD_LIBRARY_PATH=/app/cyclonedds/install/lib:${LD_LIBRARY_PATH} \
    PKG_CONFIG_PATH=/app/cyclonedds/install/lib/pkgconfig:${PKG_CONFIG_PATH}

# Project
WORKDIR /app/OM1
COPY . .
RUN git submodule update --init --recursive

# Create a dedicated virtualenv and sync deps (reads pyproject.toml)
ENV UV_PROJECT_ENV=/app/OM1/.venv
RUN uv sync --extra dds

# Entrypoint waits for network and runs OM1
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'set -e' >> /entrypoint.sh && \
    echo 'until ping -c1 -W1 8.8.8.8 >/dev/null 2>&1; do' >> /entrypoint.sh && \
    echo '  echo "Waiting for internet connection..."' >> /entrypoint.sh && \
    echo '  sleep 2' >> /entrypoint.sh && \
    echo 'done' >> /entrypoint.sh && \
    echo 'echo "Internet connected. Starting main command..."' >> /entrypoint.sh && \
    echo 'exec uv run src/run.py "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["spot"]
