## Docker image

The OM1-ros2-sdk is provided as a Docker image for easy setup.
```bash
git clone https://github.com/OpenMind/OM1-ros2-sdk.git
```
```bash
cd OM1-ros2-sdk
docker-compose up orchestrator -d --no-build
docker-compose up om1_sensor -d --no-build
docker-compose up watchdog -d --no-build
docker-compose up zenoh-bridge -d --no-build
```

The docker images are also available at Docker Hub.

- [v1.0.1](https://hub.docker.com/layers/openmindagi/om1_ros2_sdk/v1.0.1)
- [v1.0.1-beta.3](https://hub.docker.com/layers/openmindagi/om1_ros2_sdk/v1.0.1-beta.3)
- [v1.0.1-beta.2](https://hub.docker.com/layers/openmindagi/om1_ros2_sdk/v1.0.1-beta.2)
- [v1.0.1-beta.1](https://hub.docker.com/layers/openmindagi/om1_ros2_sdk/v1.0.1-beta.1)
- [v1.0.0](https://hub.docker.com/layers/openmindagi/unitree_sdk/v1.0.0)
- [v1.0.0-beta.3](https://hub.docker.com/layers/openmindagi/unitree_sdk/v1.0.0-beta.3)
- [v1.0.0-beta.2](https://hub.docker.com/layers/openmindagi/unitree_go2_sdk/v1.0.0-beta.2)
- [v1.0.0-beta.1](https://hub.docker.com/layers/openmindagi/unitree_go2_sdk/v1.0.0-beta.1)

For more technical details, please refer to the [docs](https://docs.openmind.com/full_autonomy_guidelines/architecture_overview).
