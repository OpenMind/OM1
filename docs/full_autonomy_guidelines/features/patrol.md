---
title: Patrol
description: "Run an autonomous patrol between waypoints along a route graph."
icon: shield-halved
---

Patrol turns navigation into a routine: the robot loops between the waypoints of a [route graph](maps-routes-locations.md), over and over, without anyone driving it. It's the backbone of monitoring and inspection deployments. If you've set up [auto-charging](auto-charging.md), a patrol becomes genuinely hands-off — when the battery runs low the robot docks, tops up, and picks the route back up where it left off.

Patrol is a **Go2-only** feature today, and it runs on top of Nav2 — so Nav2 needs to be up on the map first, with a route graph saved for it.

## Running a patrol

Start it with the map and route:

```bash
curl -X POST http://<robot>:5000/start/patrol \
  -H 'Content-Type: application/json' \
  -d '{"map_name": "office", "route_name": "patrol_route_1"}'
```

Pause and resume as needed:

```bash
curl -X POST http://<robot>:5000/pause/patrol  -H 'Content-Type: application/json' -d '{}'
curl -X POST http://<robot>:5000/resume/patrol -H 'Content-Type: application/json' -d '{}'
```

And stop when you're done:

```bash
curl -X POST http://<robot>:5000/stop/patrol -H 'Content-Type: application/json' -d '{}'
```

`GET /status` tells you what's running at any time — look at `patrol_status` and `current_patrol` (which map and route are loaded). Note the endpoint pattern is `<verb>/patrol` (`start`, `stop`, `pause`, `resume`), not `patrol/<verb>`.

## Parameters

`POST /start/patrol`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | yes | Map to patrol |
| `route_name` | string | yes | Route graph (GeoJSON) to follow |
| `launch_file` | string | no | Custom launch file (default `go2_patrol_launch.py`) |

## If something goes wrong

- **`400` starting** — not a Go2, Nav2 isn't running, a patrol is already going, or you left out `map_name`/`route_name`.
- **`400` route missing** — save the route first with `POST /maps/route/save`.
- **`400` on pause/resume** — nothing is patrolling.

For system endpoints like `GET /status` and base control, see the [Autonomy API Overview](../api_endpoints.md).
