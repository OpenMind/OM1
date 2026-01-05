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
    iputils-ping \
    curl \
    pkg-config \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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

RUN cp -r config config_defaults

RUN uv venv /app/OM1/.venv && \
    uv pip install -r pyproject.toml --extra dds

ENV VIRTUAL_ENV=/app/OM1/.venv
ENV PATH="/app/OM1/.venv/bin:$PATH"

RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'set -e' >> /entrypoint.sh && \
    echo 'cp -r /app/OM1/config_defaults/* /app/OM1/config/ 2>/dev/null || true' >> /entrypoint.sh && \
    echo 'until ping -c1 -W1 8.8.8.8 >/dev/null 2>&1; do' >> /entrypoint.sh && \
    echo '  echo "Waiting for internet connection..."' >> /entrypoint.sh && \
    echo '  sleep 2' >> /entrypoint.sh && \
    echo 'done' >> /entrypoint.sh && \
    echo 'echo "Internet connected. Checking audio system..."' >> /entrypoint.sh && \
    echo 'if ! pactl info >/dev/null 2>&1; then' >> /entrypoint.sh && \
    echo '  echo "ERROR: PulseAudio connection failed. Exiting container for restart..."' >> /entrypoint.sh && \
    echo '  exit 1' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo 'echo "PulseAudio connected successfully."' >> /entrypoint.sh && \
    echo 'if ! pactl list sinks | grep -q "default_output_aec" 2>/dev/null; then' >> /entrypoint.sh && \
    echo '  echo "ERROR: Audio device default_output_aec not found. Exiting container for restart..."' >> /entrypoint.sh && \
    echo '  echo "Available audio sinks:"' >> /entrypoint.sh && \
    echo '  pactl list short sinks 2>/dev/null || echo "No sinks available"' >> /entrypoint.sh && \
    echo '  exit 1' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo 'echo "Audio device default_output_aec is ready."' >> /entrypoint.sh && \
    echo 'echo "Starting main command..."' >> /entrypoint.sh && \
    echo 'if [ -f "/app/OM1/config/memory/.runtime.json5" ]; then' >> /entrypoint.sh && \
    echo '  exec python src/run.py' >> /entrypoint.sh && \
    echo 'else' >> /entrypoint.sh && \
    echo '  exec python src/run.py "$@"' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["spot"]
