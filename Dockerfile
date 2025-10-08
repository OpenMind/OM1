FROM python:3.10-slim

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
    libasound2 \
    libasound2-data \
    libasound2-plugins \
    libpulse0 \
    alsa-utils \
    alsa-topology-conf \
    alsa-ucm-conf \
    pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y curl pkg-config libssl-dev

RUN python3 -m pip install --upgrade pip

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN mkdir -p /etc/alsa && \
    ln -snf /usr/share/alsa/alsa.conf.d /etc/alsa/conf.d

RUN printf '%s\n' \
  'pcm.!default { type pulse }' \
  'ctl.!default { type pulse }' \
  > /etc/asound.conf

WORKDIR /app
RUN git clone --branch releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds
WORKDIR /app/cyclonedds/build
RUN cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=ON \
 && cmake --build . --target install

ENV CYCLONEDDS_HOME=/app/cyclonedds/install \
    CMAKE_PREFIX_PATH=/app/cyclonedds/install

WORKDIR /app/OM1
COPY . .
RUN git submodule update --init --recursive

RUN uv venv /app/OM1/.venv && \
    uv pip install -r pyproject.toml --extra dds

RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'set -e' >> /entrypoint.sh && \
    echo 'exec uv run src/run.py "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["spot"]
