FROM eclipse/zenoh-bridge-ros2dds:1.2.1

# latest is currently 1.2.1
# this image contains everything we need

# EXPOSE 7445/tcp
# EXPOSE 7446/udp
# EXPOSE 7447/tcp
# EXPOSE 8000/tcp

# let's overwrite the eclipse default entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x entrypoint.sh

WORKDIR /app

COPY config_create3.json5 /app/config_create3.json5
COPY config_rpi.json5 /app/config_rpi.json5

COPY run_zenoh_bridge_create3.sh /app/run_zenoh_bridge_create3.sh
RUN chmod +x /app/run_zenoh_bridge_create3.sh

COPY run_zenoh_bridge_rpi.sh /app/run_zenoh_bridge_rpi.sh
RUN chmod +x /app/run_zenoh_bridge_rpi.sh
