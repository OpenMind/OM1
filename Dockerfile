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

WORKDIR /opt
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
WORKDIR /opt/cyclonedds
RUN mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. \
    && cmake --build . \
    && cmake --build . --target install
ENV CYCLONEDDS_HOME=/usr/local
ENV CMAKE_PREFIX_PATH=/usr/local
ENV CYCLONEDDS_URI='<CycloneDDS> <Domain> <General> <Interfaces> <NetworkInterface name="en0" priority="default" multicast="default" /> </Interfaces> </General> <Discovery> <EnableTopicDiscoveryEndpoints>true</EnableTopicDiscoveryEndpoints> </Discovery> </Domain> </CycloneDDS>'
RUN useradd -ms /bin/bash apprunner && \
    usermod -a -G audio,video apprunner
USER apprunner
WORKDIR /home/apprunner/OM1

COPY --chown=apprunner:apprunner . .

RUN uv venv && \
    uv pip install -r pyproject.toml --extra dds
    
ENTRYPOINT ["uv", "run", "src/run.py"]
CMD ["spot"]















