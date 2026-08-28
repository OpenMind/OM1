---
title: Autonomy Features
description: "Integration guides for each OM1 autonomy feature."
icon: layer-group
---

# Autonomy Features

Hands-on guides for the autonomy features you drive through the OM1 API — mapping, navigation, patrol, charging, and the rest. Each page walks through what the feature does, when to use it, and the exact calls to make, with parameters and troubleshooting alongside.

If you're just getting oriented, the [Cloud Simulator](../../simulators/cloud-isaac-sim.md) is the fastest way to see this working without hardware. These guides assume you've done that at least once.

## Control

- [Machine Teleops](machine-teleops.md) — the portal hub: drive a robot, watch its cameras, and run its autonomy

## Mapping & navigation

- [Mapping & SLAM](mapping-slam.md) — build 2D and 3D maps
- [Hybrid Localisation](../localization.md) — how the robot tracks where it is
- [Navigation (Nav2)](navigation.md) — autonomous point-to-point on a 2D map
- [3D Map Navigation](3d-map-navigation.md) — localize and plan on a point-cloud map
- [Frontier Exploration](frontier-exploration.md) — explore an unknown space on its own

## Autonomy behaviors

- [Patrol](patrol.md) — loop a route between waypoints
- [Auto Charging](auto-charging.md) — dock, charge, and resume
- [Obstacle Avoidance](obstacle-avoidance.md) — steer around what isn't on the map
- [Person Following](person-following.md) — follow a person

## Data & memory

- [Maps, Routes & Locations](maps-routes-locations.md) — manage what the robot navigates on
- [Memory Sync](memory-sync.md) — persist and share the agent's memory via the cloud

## Monitoring & media

- [Alerts](alerts.md) — real-time alerts for battery, charging, and docking events
- [Video Recording](video-recording.md) — capture footage for later review
- [Face Detection & Anonymization](face-detection-anonymization.md) — privacy-aware perception

---

System-wide endpoints and a full feature-endpoint index live in the [Autonomy API Overview](../api_endpoints.md).
