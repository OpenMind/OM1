FROM golang:1.26-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    curl \
    unzip \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN make build

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
    libportaudio2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir -p /etc/alsa && \
    ln -snf /usr/share/alsa/alsa.conf.d /etc/alsa/conf.d

RUN printf '%s\n' \
  'pcm.!default { type pulse }' \
  'ctl.!default { type pulse }' \
  > /etc/asound.conf

RUN if ! grep -q 'mdns4_minimal' /etc/nsswitch.conf; then \
      sed -i 's/^\(hosts:[[:space:]]*files\)\(.*\)$/\1 mdns4_minimal [NOTFOUND=return]\2/' /etc/nsswitch.conf; \
    fi

WORKDIR /app/OM1

COPY --from=builder /app/build/om1 /usr/local/bin/om1
COPY --from=builder /app/.zenoh-c/lib/ /usr/local/lib/
RUN ldconfig

COPY --from=builder /app/config ./config
COPY --from=builder /app/knowledge_base ./knowledge_base

RUN cp -r config config_defaults

COPY <<'EOF' /entrypoint.sh
#!/bin/bash
set -e

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
OM1_COMMAND="${OM1_COMMAND:-${1:-conversation}}"
exec /usr/local/bin/om1 -config "${OM1_COMMAND}" ${OM1_EXTRA_ARGS}
EOF

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["conversation"]
