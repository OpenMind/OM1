---
title: Docker Images
icon: docker
---

The OM1-OTA is provided as a Docker image for easy setup.
```bash
git clone https://github.com/OpenMind/OM1-OTA
```

```bash
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

For more technical details, please refer to the [docs](https://docs.openmind.com/full_autonomy_guidelines/ota_setup).
