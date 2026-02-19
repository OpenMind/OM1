---
title: OM1 System Setup Beta Release
description: "v1.0.1-beta.1"
icon: rectangle-beta
---

## Features

### [v1.0.1-beta.1](https://github.com/OpenMind/OM1-system-setup/releases/tag/v1.0.1-beta.1)
- Improved DockerManager to treat missing containers as stopped, logging this as info and adding them to stopped_services.
- Introduced environment variable management for Docker services to simplify configuration and deployment.

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

The docker images are also available at Docker Hub [OTA](https://hub.docker.com/layers/openmindagi/ota/v1.0.1-beta.1) and [om1_monitor](https://hub.docker.com/layers/openmindagi/om1_monitor/v1.0.1-beta.1).

For more technical details, please refer to the [docs](https://docs.openmind.org/full_autonomy_guidelines/om1_system_setup).
