---
title: Auto Charging
description: "Autonomous docking and charging, with per-map charger locations."
icon: battery-full
---

For a robot that's meant to run all day, someone plugging it in defeats the purpose. Auto-charging lets it take itself to the dock and charge — and because you can save a charger location per map, it knows where the dock is in each space it works. Paired with [patrol](patrol.md), it closes the loop: patrol until low, dock, charge, resume.

Like patrol, this is **Go2-only** and needs Nav2 running.

## Docking and monitoring

Send the robot to dock:

```bash
curl -X POST http://<robot>:5000/charging/dock -H 'Content-Type: application/json' -d '{}'
```

Watch the battery and docking state:

```bash
curl http://<robot>:5000/charging/status
```

You get back `is_charging`, `battery_soc` (%), `battery_current` (mA — negative discharging, positive charging), `battery_voltage`, `battery_temperature`, `dock_process_running`, and `charging_confirmation_pending` (charging detected but not yet confirmed).

Abort or undock:

```bash
curl -X POST http://<robot>:5000/charging/stop -H 'Content-Type: application/json' -d '{}'
```

## Telling the robot where the dock is

Read the current charger waypoints, optionally for a specific map:

```bash
curl "http://<robot>:5000/charging/location?map_name=office"
```

OM1 resolves the location in order — a **per-map override**, then a **global waypoint file**, then **built-in defaults** — and the `source` field in the response tells you which one it used.

If the robot docks in the wrong place, save a per-map override with a `predock` staging pose and the `charger` pose:

```bash
curl -X POST http://<robot>:5000/charging/location/save \
  -H 'Content-Type: application/json' \
  -d '{"map_name": "office",
       "predock": {"position": {"x": 0.5, "y": -2.0, "z": 0}, "orientation": {"x":0,"y":0,"z":-0.707,"w":0.707}},
       "charger": {"position": {"x": 0.5, "y": -1.0, "z": 0}, "orientation": {"x":0,"y":0,"z":-0.707,"w":0.707}}}'
```

## If something goes wrong

- **`400` on dock** — not a Go2, Nav2 isn't running, or it's already charging/docking.
- **`400` on save** — a field is missing, a pose failed validation, or the `map_name` is invalid or doesn't exist.
- **Wrong dock spot** — save a per-map override as above.

For system endpoints like `GET /status` and base control, see the [Autonomy API Overview](../api_endpoints.md).
