## Docker image

The OM1-system-setup is provided as a Docker image for easy setup.
```bash
git clone https://github.com/OpenMind/OM1-system-setup
```

```bash
    cd OM1-system-setup
    cd WIFI
    docker-compose up -d om1_monitor

    cd ..
    cd OTA
    docker-compose up -d ota_agent
    docker-compose up -d ota_updater
```

The docker images are also available at Docker Hub.

**OTA**

- [v1.0.2](https://hub.docker.com/layers/openmindagi/ota/v1.0.2)
- [v1.0.1](https://hub.docker.com/layers/openmindagi/ota/v1.0.1)
- [v1.0.0](https://hub.docker.com/layers/openmindagi/ota/v1.0.0)
- [v1.0.0-beta.1](https://hub.docker.com/layers/openmindagi/ota/v1.0.0-beta.1)

**om1_monitor**

- [v1.0.2](https://hub.docker.com/layers/openmindagi/om1_monitor/v1.0.2)
- [v1.0.1](https://hub.docker.com/layers/openmindagi/om1_monitor/v1.0.1)
- [v1.0.0](https://hub.docker.com/layers/openmindagi/om1_monitor/v1.0.0)
- [v1.0.0-beta.1](https://hub.docker.com/layers/openmindagi/om1_monitor/v1.0.0-beta.1)

For more technical details, please refer to the [docs](https://docs.openmind.com/full_autonomy_guidelines/ota_setup).
