---
title: Mapping & SLAM
description: "Build 2D and 3D maps of an environment with the OM1 autonomy stack."
icon: map
---

Before a robot can navigate on its own, it needs a map. SLAM — simultaneous localization and mapping — is how OM1 builds one: you drive the robot through a space, and from its LiDAR and depth sensors it assembles a map while keeping track of where it is inside it.

There are two ways to map, both started with a single API call:

- **2D SLAM** for flat, single-floor spaces. It's the quickest path to autonomous navigation and produces an occupancy grid (`.pgm` / `.yaml`).
- **3D SLAM** for anything with ramps or multiple levels. It captures a full point cloud (`.pcd`) and, from the same data, also rasterizes a 2D grid — so one 3D run gives you both map types, sharing a single world origin.

If you just want to watch mapping happen, the [Cloud Simulator](../../simulators/cloud-isaac-sim.md) walks through it end to end without hardware. The rest of this page is the hands-on version for a real robot.

## Before you start

- The robot is online and its Orchestrator is reachable at `http://<robot>:5000`.
- **Nav2 is stopped** — SLAM and Nav2 can't run at the same time.
- For 3D, the robot actually has a 3D stack. Check `slam_3d_supported` in `GET /status` first; not every platform has it.

## Building a map

Kick off a run — 2D:

```bash
curl -X POST http://<robot>:5000/start/slam/2d -H 'Content-Type: application/json' -d '{}'
```

…or 3D:

```bash
curl -X POST http://<robot>:5000/start/slam/3d -H 'Content-Type: application/json' -d '{}'
```

Starting 2D SLAM also turns on [frontier exploration](frontier-exploration.md), so the robot begins covering the space on its own — or you can just teleoperate it. Watch the map fill in from the portal's live view, and pause or resume exploration with `/explore/stop` and `/explore/resume` whenever you need to take manual control.

Drove somewhere worth remembering? Save it as a named waypoint while SLAM is still running:

```bash
curl -X POST http://<robot>:5000/maps/locations/add/slam \
  -H 'Content-Type: application/json' \
  -d '{"map_name": "office", "label": "reception", "description": "Front desk"}'
```

When the map looks complete, save it:

```bash
curl -X POST http://<robot>:5000/maps/save -H 'Content-Type: application/json' -d '{"map_name": "office"}'
```

One call writes every artifact the current mode can produce into a single folder — there's no partial option. One thing to watch: check the `status` field in the response, not just the HTTP code. A `partial_success` means one artifact saved and another didn't, with the details in `errors`.

Then stop SLAM:

```bash
curl -X POST http://<robot>:5000/stop/slam -H 'Content-Type: application/json' -d '{}'
```

Save before you stop — stopping tears the map down.

## Parameters

`POST /start/slam/2d`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `launch_file` | string | `slam_launch.py` | Custom SLAM launch file |
| `map_yaml` | string | — | Path to an existing map to continue mapping from |

`POST /start/slam/3d`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `launch_file` | string | `slam_launch.py` | Dispatched to the 3D stack via `slam_mode:=3d` |
| `rviz` | bool | `false` | Launch RViz2 with the 3D SLAM view |

`POST /maps/save`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | yes | Map name — no `/`, `\`, `..`, or spaces |
| `map_directory` | string | no | Custom storage directory |

## What you get

A saved map is a folder under `maps/<map_name>/`. What's in it depends on how you mapped:

- **2D:** `<name>.pgm`, `<name>.yaml`, `<name>.posegraph`, `<name>.data`, `<name>_vpr.npz`
- **3D:** all of the above, plus `<name>.pcd` (full resolution) and `<name>_downsampled.pcd` (the one uploaded to the cloud)

A folder with only a grid can't be used for [3D map navigation](3d-map-navigation.md), and one with only a point cloud can't be used by [Nav2](navigation.md) — so what a map supports follows from the mode that made it.

## If something goes wrong

- **`400` when starting SLAM** — Nav2 is running, or SLAM already is. Check `GET /status` and stop the other one.
- **`400` on `/start/slam/3d`** — the robot type has no 3D stack. Confirm `slam_3d_supported: true`.
- **`partial_success` on save** — one artifact type failed; the `errors` field says which. Whatever saved is still uploaded.

For system endpoints like `GET /status` and base control, see the [Autonomy API Overview](../api_endpoints.md).
