---
title: Frontier Exploration
description: "Autonomously explore and map an unknown space during 2D SLAM."
icon: compass
---

Mapping a space by hand — teleoperating the robot into every corner — is tedious. Frontier exploration does it for you: the robot heads for the edges between mapped and unmapped space (the "frontiers") and keeps going until there's nothing left to discover.

It isn't a separate mode you start. It comes on automatically with **2D SLAM** — the moment you call `POST /start/slam/2d`, the robot starts exploring — and it shuts down when you stop SLAM. (3D SLAM doesn't use it.)

While SLAM is up, you can hand control back and forth. Pause exploration to teleoperate a tricky spot, then resume:

```bash
curl -X POST http://<robot>:5000/explore/stop   -H 'Content-Type: application/json' -d '{}'
curl -X POST http://<robot>:5000/explore/resume -H 'Content-Type: application/json' -d '{}'
```

Check where it's at:

```bash
curl http://<robot>:5000/explore/status
```

The `message` field is a JSON-encoded string — decode it for `running`, `paused`, `complete`, an `info` string, and `status_received` (whether the explorer has reported in yet). Once `complete` is true, save the map with `POST /maps/save` before you stop SLAM.

If pause and resume seem to do nothing, 2D SLAM probably isn't running — the explorer only exists while it is. See [Mapping & SLAM](mapping-slam.md) for the full mapping flow, and the [Autonomy API Overview](../api_endpoints.md) for endpoint details.
