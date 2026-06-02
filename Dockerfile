# syntax=docker/dockerfile:1

# ---------- Build stage ----------
FROM golang:1.22-bookworm AS builder

# Tools needed to fetch/extract zenoh-c, build FAISS, and build the cgo binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    cmake \
    curl \
    unzip \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Build and install FAISS C library.
RUN git clone --depth 1 --branch v1.9.0 https://github.com/facebookresearch/faiss.git /tmp/faiss \
    && cmake -B /tmp/faiss/build -S /tmp/faiss \
        -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
        -DFAISS_ENABLE_C_API=ON \
        -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build /tmp/faiss/build -j$(nproc) \
    && cmake --install /tmp/faiss/build \
    && ldconfig \
    && rm -rf /tmp/faiss

WORKDIR /app

# Cache module downloads first.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Fetch the zenoh-c native library and build the binary. The Makefile
# handles platform detection, the zenoh-c download, and the cgo flags.
RUN make build

# ---------- Runtime stage ----------
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libasound2 \
    libasound2-data \
    libasound2-plugins \
    libpulse0 \
    alsa-utils \
    alsa-topology-conf \
    alsa-ucm-conf \
    pulseaudio-utils \
    libv4l-0 \
    libhidapi-hidraw0 \
    iputils-ping \
    libnss-mdns \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ALSA -> PulseAudio routing.
RUN mkdir -p /etc/alsa && \
    ln -snf /usr/share/alsa/alsa.conf.d /etc/alsa/conf.d

RUN printf '%s\n' \
  'pcm.!default { type pulse }' \
  'ctl.!default { type pulse }' \
  > /etc/asound.conf

# Enable mDNS host resolution.
RUN if ! grep -q 'mdns4_minimal' /etc/nsswitch.conf; then \
      sed -i 's/^\(hosts:[[:space:]]*files\)\(.*\)$/\1 mdns4_minimal [NOTFOUND=return]\2/' /etc/nsswitch.conf; \
    fi

WORKDIR /app/OM1

# Binary and the zenoh-c shared library it links against at runtime.
COPY --from=builder /app/build/om1 /usr/local/bin/om1
COPY --from=builder /app/.zenoh-c/lib/ /usr/local/lib/
COPY --from=builder /usr/local/lib/libfaiss* /usr/local/lib/
RUN ldconfig

# Runtime assets.
COPY --from=builder /app/config ./config
COPY --from=builder /app/knowledge_base ./knowledge_base

# Keep a pristine copy of the bundled configs so a mounted volume can be
# re-seeded at startup.
RUN cp -r config config_defaults

COPY <<'EOF' /entrypoint.sh
#!/bin/bash
set -e

# Re-seed any missing default configs (no-op if the volume already has them).
cp -rn /app/OM1/config_defaults/* /app/OM1/config/ 2>/dev/null || true

if [ "${OM1_SKIP_INTERNET_CHECK}" = "true" ]; then
  echo "Skipping internet connectivity check."
else
  until ping -c1 -W1 8.8.8.8 >/dev/null 2>&1; do
    echo "Waiting for internet connection..."
    sleep 2
  done
  echo "Internet connected."
fi

if [ "${OM1_SKIP_AUDIO_CHECK}" = "true" ]; then
  echo "Skipping audio system check."
else
  echo "Checking audio system..."
  if ! pactl info >/dev/null 2>&1; then
    echo "ERROR: PulseAudio connection failed. Exiting container for restart..."
    exit 1
  fi
  echo "PulseAudio connected successfully."
  if ! pactl list sinks | grep -q "default_output_aec" 2>/dev/null; then
    echo "ERROR: Audio device default_output_aec not found. Exiting container for restart..."
    echo "Available audio sinks:"
    pactl list short sinks 2>/dev/null || echo "No sinks available"
    exit 1
  fi
  echo "Audio device default_output_aec is ready."
fi

echo "Starting OM1..."
CONFIG="${OM1_CONFIG:-${1:-conversation}}"
exec /usr/local/bin/om1 -config "${CONFIG}" ${OM1_EXTRA_ARGS}
EOF

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["conversation"]
