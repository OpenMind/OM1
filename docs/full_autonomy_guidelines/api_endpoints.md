---
title: Autonomy API Overview
description: "The OM1 autonomy REST APIs — system endpoints, live monitoring, and where each feature's endpoints live."
icon: link
---

The OM1 ROS2 SDK exposes two REST APIs for controlling and monitoring an autonomous robot:

| API | Port | What it's for |
|-----|------|---------------|
| **Orchestrator API** | `5000` | Start/stop autonomy processes and manage maps, routes, and locations |
| **Nav2 API** | `5001` | Read live navigation and localization data |

> Orchestrator endpoints return a JSON object with a `status` field (`"success"`, `"partial_success"`, or `"error"`). Several `GET` endpoints return their payload as a **JSON-encoded string** in the `message` field — decode `message` a second time to read the fields.

This page covers the **system-wide endpoints** that aren't tied to a single feature. Each feature's own endpoints are documented in that feature's guide — see the [index below](#feature-endpoints).

## System status

### `GET /status`

The one call to check what's running and the battery state before starting or stopping anything. The `message` field is a JSON-encoded string:

```json
{
  "status": "success",
  "message": "{\"slam_status\": \"stopped\", \"nav2_status\": \"running\", \"nav3d_status\": \"stopped\", \"base_control_status\": \"running\", \"patrol_status\": \"stopped\", \"charging_dock_status\": \"stopped\", \"is_charging\": false, \"battery_soc\": 87.0, \"battery_current\": -500.0, \"battery_voltage\": 32.4, \"battery_temperature\": 25.0, \"current_patrol\": {\"map_name\": \"kitchen\", \"route_name\": null}, \"robot_type\": \"go2\", \"slam_3d_supported\": true}"
}
```

**Decoded `message` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `slam_status`, `nav2_status`, `nav3d_status`, `base_control_status`, `patrol_status`, `charging_dock_status` | `"running"` \| `"stopped"` | Per-process status |
| `is_charging` | `bool` | Whether the robot is charging |
| `battery_soc` | `float` | Battery state of charge (%) |
| `battery_current` | `float` | mA; negative = discharging, positive = charging |
| `battery_voltage` | `float` | Pack voltage (V) |
| `battery_temperature` | `float` | °C |
| `current_patrol` | `object` \| `null` | Map (and route, if any) currently loaded, or `null`. Precedence: patrol > Nav2 > 3D nav; only `map_name` is guaranteed |
| `robot_type` | `str` | e.g. `"go2"`, `"g1"`, `"tron"`, `"m20"` |
| `slam_3d_supported` | `bool` | Whether this robot has a 3D SLAM stack — use this to decide whether to offer 3D SLAM |

## Base control

Base control is the low-level motor layer; it must be running for the robot to move, and it's a prerequisite rather than a feature of its own.

```bash
curl -X POST http://<robot>:5000/start/base_control -H 'Content-Type: application/json' -d '{}'
curl -X POST http://<robot>:5000/stop/base_control  -H 'Content-Type: application/json' -d '{}'
```

`/start/base_control` accepts an optional `launch_file` (default `base_control_launch.py`), and returns `400` if SLAM or Nav2 is already running.

## Live monitoring (Nav2 API, `:5001`)

Read-only endpoints for dashboards and health checks. (Sending navigation goals lives with [Navigation](features/navigation.md).)

| Endpoint | Returns |
|----------|---------|
| `GET /api/status` | API health — `{ "status": "OK", "message": "..." }` |
| `GET /api/pose` | Current pose in the map frame with a 36-element `covariance` array |
| `GET /api/amcl_variance` | Localization uncertainty — `x_uncertainty`, `y_uncertainty` (m), `yaw_uncertainty` (deg) |
| `GET /api/map` | The occupancy grid — `map_metadata` (resolution, width, height, origin) plus a `data` array (`-1` unknown, `0` free, `100` occupied) |

## Feature endpoints

Every feature-specific endpoint is documented — with how, when, and why to use it — in that feature's guide:

| Feature | Endpoints | Guide |
|---------|-----------|-------|
| Mapping & SLAM | `/start/slam/2d`, `/start/slam/3d`, `/stop/slam`, `/maps/save` | [Mapping & SLAM](features/mapping-slam.md) |
| Navigation | `/start/nav2`, `/stop/nav2`, `/api/move_to_pose`, `/api/nav2_status` | [Navigation (Nav2)](features/navigation.md) |
| 3D Map Navigation | `/start/nav3d`, `/stop/nav3d` | [3D Map Navigation](features/3d-map-navigation.md) |
| Frontier Exploration | `/explore/stop`, `/explore/resume`, `/explore/status` | [Frontier Exploration](features/frontier-exploration.md) |
| Patrol | `/start/patrol`, `/stop/patrol`, `/pause/patrol`, `/resume/patrol` | [Patrol](features/patrol.md) |
| Auto Charging | `/charging/dock`, `/charging/stop`, `/charging/status`, `/charging/location`, `/charging/location/save` | [Auto Charging](features/auto-charging.md) |
| Maps, Routes & Locations | `/maps/list`, `/maps/delete`, `/maps/route/save`, `/maps/locations/*` | [Maps, Routes & Locations](features/maps-routes-locations.md) |
