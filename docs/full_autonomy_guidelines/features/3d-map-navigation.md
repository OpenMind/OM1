---
title: 3D Map Navigation
description: "Localize and plan on a point-cloud map built with 3D SLAM."
icon: cube
---

When a flat 2D grid isn't enough — think ramps, split levels, or anything where height matters — you navigate against the **point-cloud map** you captured with [3D SLAM](mapping-slam.md) instead. 3D map navigation localizes the robot against that `.pcd` and plans paths over it.

Two pieces come up together: **ICP localization**, which matches the live LiDAR against the saved point cloud to track where the robot is, and the **PCT planner**, which plans a route from there to a goal. One thing to design around: this **localizes and plans only — it doesn't drive the robot**. A separate motion controller consumes the planned path, so pair it accordingly.

## In the portal

The [OpenMind portal](https://portal.openmind.com) lets you load a 3D (point-cloud) map and set navigation goals against it from the machine's autonomy view, the same way you would for a 2D map.

## Before you start

- You have a map with a saved **`.pcd`** — meaning it was saved while 3D SLAM was running.
- The robot type supports 3D (`slam_3d_supported` in `GET /status`).
- **SLAM and Nav2 are both stopped.**

## Running it

Start it against a 3D map:

```bash
curl -X POST http://<robot>:5000/start/nav3d \
  -H 'Content-Type: application/json' \
  -d '{"map_name": "warehouse"}'
```

The planner listens for goal poses on its goal topic (`/goal_pose` by default) and publishes the path it finds. Send goals from your integration or the portal. When you're done:

```bash
curl -X POST http://<robot>:5000/stop/nav3d -H 'Content-Type: application/json' -d '{}'
```

## Parameters

`POST /start/nav3d`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `map_name` | string | yes | — | A map with a saved `.pcd` |
| `scene` | string | no | `Isaacsim` | Tomography scene config |
| `goal_topic` | string | no | `/goal_pose` | Topic the planner listens on for goals |
| `goal_layer` | int | no | `0` | Tomogram layer index for the goal |

## If something goes wrong

- **`400`** — SLAM or Nav2 is running, or 3D nav is already up. Stop the other one first.
- **`404`** — the map has no `.pcd`; it wasn't saved during 3D SLAM.
- **`400` unsupported** — the robot type has no 3D stack.

For system endpoints like `GET /status` and base control, see the [Autonomy API Overview](../api_endpoints.md).
