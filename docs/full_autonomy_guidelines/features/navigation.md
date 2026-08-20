---
title: Navigation (Nav2)
description: "Autonomous point-to-point navigation on a saved 2D map."
icon: route
---

Once you have a [map](mapping-slam.md), navigation is what makes the robot useful on its own: give it a destination and it plans a path there, drives, and steers around whatever's in the way. Under the hood it's the Nav2 stack handling planning, obstacle avoidance, and localization.

There are two APIs involved. You start and stop the Nav2 stack through the Orchestrator (`:5000`), then send goals and watch their progress through the Nav2 API (`:5001`). If you want the robot to stick to fixed paths rather than plan freely, point it at a [route graph](maps-routes-locations.md).

## In the portal

From the [OpenMind portal](https://portal.openmind.com) you can send the robot somewhere by clicking a point on the map or picking a saved [location](maps-routes-locations.md) — no API calls needed. The map view also shows the robot's live pose as it drives.

![Select a map for navigation in the portal](../../.gitbook/assets/full-autonomy-assets/select_location_to_navigate.png)

## Before you start

- You have a **saved 2D map** (from either 2D or 3D SLAM).
- **SLAM is stopped** — Nav2 and SLAM can't run together.
- If you're using a route graph, it's already saved for that map.

## Driving to a goal

Bring up Nav2 on the map you want to use:

```bash
curl -X POST http://<robot>:5000/start/nav2 \
  -H 'Content-Type: application/json' \
  -d '{"map_name": "office"}'
```

Add `"route_name": "patrol_route_1"` to constrain it to a route graph.

The robot localizes itself in the map, and you can now send it a goal pose (in the map frame) through the Nav2 API:

```bash
curl -X POST http://<robot>:5001/api/move_to_pose \
  -H 'Content-Type: application/json' \
  -d '{"position": {"x": 1.0, "y": 2.0, "z": 0.0},
       "orientation": {"x": 0, "y": 0, "z": 0, "w": 1.0}}'
```

That call returns immediately — navigation is asynchronous — so poll for progress:

```bash
curl http://<robot>:5001/api/nav2_status
```

Each active goal reports a `status`: `ACCEPTED`, `EXECUTING`, `SUCCEEDED`, `CANCELING`, `CANCELED`, or `ABORTED`. The same API also exposes the live robot pose (`/api/pose`), localization confidence (`/api/amcl_variance`), and the current occupancy grid (`/api/map`) — handy for a dashboard.

When you're done:

```bash
curl -X POST http://<robot>:5000/stop/nav2 -H 'Content-Type: application/json' -d '{}'
```

## Parameters

`POST /start/nav2`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | yes | Saved map to navigate on |
| `launch_file` | string | no | Custom Nav2 launch file (default `nav2_launch.py`) |
| `route_name` | string | no | Route graph for graph-constrained navigation |

`POST /api/move_to_pose` (Nav2 API, `:5001`) takes a `position` (`x, y, z` in the map frame) and an `orientation` (quaternion) — both required.

## If something goes wrong

- **`400` on `/start/nav2`** — SLAM is still running, or you left out `map_name`.
- **`400` with a route** — that route graph doesn't exist for the map; [save it](maps-routes-locations.md) first.
- **Goal `ABORTED` right away** — usually poor localization or a blocked path. Check `/api/amcl_variance` and clear the route.

For system endpoints like `GET /status` and base control, see the [Autonomy API Overview](../api_endpoints.md).
