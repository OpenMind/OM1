FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    libv4l-dev \
    ca-certificates \
    build-essential \
    cmake \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /bin/uv /usr/local/bin/
COPY --from=builder /bin/uvx /usr/local/bin/

RUN mkdir app   
WORKDIR /app
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
WORKDIR /app/cyclonedds
RUN mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. \
    && cmake --build . \
    && cmake --build . --target install
RUN mkdir /app/OM1
WORKDIR /app/OM1

COPY . /app/OM1/
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'set -e' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Source cyclonedds environment' >> /entrypoint.sh && \
    echo 'export CYCLONEDDS_HOME=/usr/local' >> /entrypoint.sh && \
    echo 'export CMAKE_PREFIX_PATH=/usr/local' >> /entrypoint.sh && \
    echo 'export CYCLONEDDS_URI="<CycloneDDS> <Domain> <General> <Interfaces> <NetworkInterface name="en0" priority="default" multicast="default" /> </Interfaces> </General> <Discovery> <EnableTopicDiscoveryEndpoints>true</EnableTopicDiscoveryEndpoints> </Discovery> </Domain> </CycloneDDS>"' >> /entrypoint.sh && \
    echo 'uv venv' >> /entrypoint.sh && \
    echo 'uv pip install -r pyproject.toml --extra dds' >> /entrypoint.sh && \
    echo 'uv run src/run.py "$@"' >> /entrypoint.sh 
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["spot"]