---
title: Premium Features API Endpoints
description: "REST API documentation for OM1 ROS2 SDK"
icon: link
---

## Overview

The OM1 ROS2 SDK provides two REST APIs for remote control and monitoring of the robot:

| API | Port | Description |
|-----|------|-------------|
| **Orchestrator API** | `5000` | System orchestration and high-level control (SLAM, Nav2, 3D map navigation, charging, patrol, maps, routes, locations) |
| **Nav2 API** | `5001` | Direct navigation control and real-time monitoring (pose, goals, localization, map) |

> **Note:** These APIs are part of the **Premium Features** available on the OpenMind autonomy stack (BrainPack). See [Premium Features](../developing/premium_features.md) for details.

> All Orchestrator endpoints return a JSON object with a `status` field (`"success"`, `"partial_success"`, or `"error"`). Several `GET` endpoints return their payload as a **JSON-encoded string** in the `message` field — you must decode `message` a second time to read the fields.

---

## Orchestrator API (Port 5000)

The Orchestrator API manages the robot's operational state: base control, SLAM (2D and 3D), Nav2, 3D map navigation, charging, patrol, frontier exploration, and map/route/location management.

### Quick Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/status` | Status of all running processes + battery |
| `POST` | `/start/base_control` | Start base control process |
| `POST` | `/stop/base_control` | Stop base control process |
| `POST` | `/start/slam/2d` | Start 2D SLAM (slam_toolbox + Nav2 + frontier explorer) |
| `POST` | `/start/slam/3d` | Start 3D SLAM (FAST-LIO + rasterized 2D grid) |
| `POST` | `/stop/slam` | Stop SLAM (2D or 3D) |
| `POST` | `/start/nav2` | Start Nav2 navigation stack (optionally with a route graph) |
| `POST` | `/stop/nav2` | Stop Nav2 navigation stack |
| `POST` | `/start/nav3d` | Start 3D map navigation (ICP localization + PCT planner) |
| `POST` | `/stop/nav3d` | Stop 3D map navigation |
| `POST` | `/charging/dock` | Start autonomous docking (Go2 only) |
| `POST` | `/charging/stop` | Stop docking process |
| `GET` | `/charging/status` | Get charging status |
| `GET` | `/charging/location` | Get charging station waypoints |
| `POST` | `/charging/location/save` | Save a per-map charging-station override |
| `POST` | `/start/patrol` | Start patrol (Go2 only) |
| `POST` | `/stop/patrol` | Stop patrol |
| `POST` | `/pause/patrol` | Pause patrol |
| `POST` | `/resume/patrol` | Resume patrol |
| `POST` | `/explore/stop` | Pause frontier exploration |
| `POST` | `/explore/resume` | Resume frontier exploration |
| `GET` | `/explore/status` | Get frontier exploration status |
| `POST` | `/maps/save` | Save the current SLAM map (all artifacts) |
| `GET` | `/maps/list` | List all saved maps |
| `POST` | `/maps/delete` | Delete a saved map |
| `POST` | `/maps/route/save` | Save a route graph (GeoJSON) to a map |
| `POST` | `/maps/locations/add` | Add a location to a map |
| `GET` | `/maps/locations/list` | List all saved locations |
| `POST` | `/maps/locations/add/slam` | Save the current position as a location |

---

### System Status

#### GET /status

Get the status of all running processes plus battery telemetry. Use this to check which services are active before starting or stopping processes. The `message` field is a **JSON-encoded string** of the full status.

**Response:**
```json
{
  "status": "success",
  "message": "{\"slam_status\": \"stopped\", \"nav2_status\": \"running\", \"nav3d_status\": \"stopped\", \"base_control_status\": \"running\", \"patrol_status\": \"stopped\", \"charging_dock_status\": \"stopped\", \"is_charging\": false, \"battery_soc\": 87.0, \"battery_current\": -500.0, \"battery_voltage\": 32.4, \"battery_temperature\": 25.0, \"current_patrol\": {\"map_name\": \"kitchen\", \"route_name\": null}, \"robot_type\": \"go2\", \"slam_3d_supported\": true}"
}
```

**Decoded `message` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `slam_status` | `"running"` \| `"stopped"` | SLAM process status |
| `nav2_status` | `"running"` \| `"stopped"` | Nav2 process status |
| `nav3d_status` | `"running"` \| `"stopped"` | 3D map navigation process status |
| `base_control_status` | `"running"` \| `"stopped"` | Base control process status |
| `patrol_status` | `"running"` \| `"stopped"` | Patrol process status |
| `charging_dock_status` | `"running"` \| `"stopped"` | Charging dock process status |
| `is_charging` | `bool` | Whether the robot is charging |
| `battery_soc` | `float` | Battery state of charge (%) |
| `battery_current` | `float` | Battery current (mA); negative = discharging, positive = charging |
| `battery_voltage` | `float` | Battery pack voltage (V) |
| `battery_temperature` | `float` | Battery temperature (°C) |
| `current_patrol` | `object` \| `null` | Map (and route, if any) currently loaded in patrol/Nav2/3D nav, or `null` when none are running |
| `robot_type` | `str` | Robot type (from the `ROBOT_TYPE` env var, e.g. `"go2"`, `"g1"`, `"tron"`, `"m20"`) |
| `slam_3d_supported` | `bool` | Whether this robot type has a FAST-LIO 3D SLAM stack. Use this to decide whether to offer 3D SLAM, rather than guessing from `robot_type` |

`current_patrol` reflects whatever is running, with precedence patrol > Nav2 > 3D map navigation. Only `map_name` is guaranteed; Nav2 or 3D nav without a route reports `route_name: null`.

---

### Base Control

Base control manages the low-level motor commands that allow the robot to move. It must be running for movement operations.

#### POST /start/base_control

Start the base control process.

**Request body (optional):**
```json
{ "launch_file": "base_control_launch.py" }
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `launch_file` | string | No | Custom launch file (default: `base_control_launch.py`) |

**Response:**
```json
{ "status": "success", "message": "Base control started" }
```

**Error conditions:** `400` if SLAM or Nav2 is already running.

#### POST /stop/base_control

Stop the base control process.

**Response:**
```json
{ "status": "success", "message": "Base control stopped" }
```

---

### SLAM

SLAM runs in one of two mutually exclusive modes (only one SLAM process at a time):

- **2D** (`/start/slam/2d`) — `slam_toolbox` + the Nav2 stack + frontier explorer. Produces a 2D occupancy grid.
- **3D** (`/start/slam/3d`) — FAST-LIO LiDAR-inertial SLAM, plus a 2D occupancy grid rasterized from the same cloud. A single 3D run produces **both** map types, and `/maps/save` writes all of them into one folder.

#### POST /start/slam/2d

Start 2D SLAM (slam_toolbox + Nav2) for mapping. This also launches the frontier explorer, which begins exploring automatically (see [Frontier Exploration](#frontier-exploration-go2-only)).

**Request body (optional):**
```json
{ "launch_file": "slam_launch.py", "map_yaml": "/path/to/existing/map.yaml" }
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `launch_file` | string | No | Custom SLAM launch file |
| `map_yaml` | string | No | Path to an existing map to continue mapping from |

**Response:**
```json
{ "status": "success", "message": "2D SLAM started" }
```

**Error conditions:**
- `400` if Nav2 is running (stop Nav2 first)
- `400` if SLAM is already running

#### POST /start/slam/3d

Start 3D SLAM (FAST-LIO). Alongside FAST-LIO the stack rasterizes a 2D occupancy grid from the same cloud, so one mapping run yields both map types (there is no switch to turn the 2D branch off). The FAST-LIO config is chosen automatically by the `use_sim` node parameter.

**Request body (optional):**
```json
{ "launch_file": "slam_launch.py", "rviz": false }
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `launch_file` | string | No | Defaults to `slam_launch.py` (dispatched to the 3D stack via `slam_mode:=3d`) |
| `rviz` | bool | No | Launch RViz2 with the FAST-LIO view (default `false`) |

**Response:**
```json
{ "status": "success", "message": "3D SLAM started" }
```

**Error conditions:**
- `400` if the robot type has no 3D SLAM stack (check `slam_3d_supported` in `GET /status`)
- `400` if Nav2 is running (stop Nav2 first)
- `400` if SLAM is already running

#### POST /stop/slam

Stop the SLAM process (2D or 3D). Save the map with `/maps/save` before stopping if you want to keep it.

**Response:**
```json
{ "status": "success", "message": "SLAM stopped" }
```

---

### Navigation (Nav2)

Nav2 is the ROS2 navigation stack for autonomous navigation on a pre-built 2D map: path planning, obstacle avoidance, and localization. It optionally supports route-graph navigation.

#### POST /start/nav2

Start the Nav2 navigation stack with a saved map.

**Request body:**
```json
{
  "map_name": "office_floor1",
  "launch_file": "nav2_launch.py",
  "route_name": "patrol_route_1"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | **Yes** | Name of the saved map to use |
| `launch_file` | string | No | Custom Nav2 launch file (default: `nav2_launch.py`) |
| `route_name` | string | No | Route graph file for graph-based navigation |

**Response:**
```json
{ "status": "success", "message": "Nav2 started" }
```

**Error conditions:**
- `400` if SLAM is running (stop SLAM first)
- `400` if `map_name` is not provided
- `400` if `route_name` is provided but the route graph file does not exist

#### POST /stop/nav2

Stop the Nav2 navigation stack (cancels any active goals).

**Response:**
```json
{ "status": "success", "message": "Nav2 stopped" }
```

---

### 3D Map Navigation

An alternative navigation path for point-cloud maps saved via `/maps/save` while 3D SLAM was running. It brings up **ICP localization** (registers live lidar against the saved `.pcd` to produce the `map -> base_link` TF) and the **PCT planner** (builds a tomogram from the `.pcd` and plans a path to a goal on `/goal_pose`, publishing to `/pct_path`).

> This only localizes and plans — it does not drive the robot. There is no controller consuming `/pct_path`, so `base_control` (or another `/cmd_vel` consumer) is left running.

#### POST /start/nav3d

Start 3D map navigation against a saved point-cloud map.

**Request body:**
```json
{
  "map_name": "office_floor1",
  "scene": "Isaacsim",
  "goal_topic": "/goal_pose",
  "goal_layer": 0
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | **Yes** | A map with a saved `.pcd` (i.e. saved while 3D SLAM was running) |
| `scene` | string | No | Tomography scene config (default `"Isaacsim"`) |
| `goal_topic` | string | No | Topic the PCT planner listens on for goals (default `"/goal_pose"`) |
| `goal_layer` | int | No | Tomogram layer index for the goal (default `0`) |

**Response:**
```json
{ "status": "success", "message": "3D map navigation started" }
```

**Error conditions:**
- `400` if the robot type has no 3D SLAM stack (check `slam_3d_supported`)
- `400` if SLAM or Nav2 is running (stop it first)
- `400` if 3D map navigation is already running
- `400` if `map_name` is not provided
- `404` if the map's `.pcd` file does not exist

#### POST /stop/nav3d

Stop 3D map navigation (PCT planner + ICP localization).

**Response:**
```json
{ "status": "success", "message": "3D map navigation stopped" }
```

**Error conditions:** `400` if 3D map navigation is not running.

---

### Charging (Go2 Only)

Autonomous docking and charging. Only available on the Unitree Go2.

#### POST /charging/dock

Start the autonomous docking sequence.

**Prerequisites:** Nav2 must be running; robot must not already be charging.

**Request body (optional):**
```json
{ "launch_file": "go2_charge_launch.py" }
```

**Response:**
```json
{ "status": "success", "message": "Charging dock process started" }
```

**Error conditions:** `400` if robot type is not Go2; `400` if Nav2 is not running; `400` if already charging or the dock process is running.

#### POST /charging/stop

Stop the charging dock process.

**Response:**
```json
{ "status": "success", "message": "Charging dock process stopped" }
```

#### GET /charging/status

Get current charging and docking status.

**Response:**
```json
{
  "is_charging": false,
  "battery_soc": 87.0,
  "battery_current": -500.0,
  "battery_voltage": 32.4,
  "battery_temperature": 25.0,
  "dock_process_running": false,
  "charging_confirmation_pending": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `is_charging` | `bool` | Whether the robot is charging |
| `battery_soc` | `float` | Battery state of charge (%) |
| `battery_current` | `float` | Battery current (mA); negative = discharging, positive = charging |
| `battery_voltage` | `float` | Battery pack voltage (V) |
| `battery_temperature` | `float` | Battery temperature (°C) |
| `dock_process_running` | `bool` | Whether the docking process is running |
| `charging_confirmation_pending` | `bool` | Whether charging is detected but not yet confirmed |

#### GET /charging/location

Get the charging station's predock and charger waypoints.

**Query parameters (optional):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `map_name` | string | Active map name. When given, a per-map override is checked first. |

Resolution order: (1) per-map override `maps/<map_name>/<map_name>.charging_station.json`, (2) global `patrol_waypoints.json` (`WAYPOINT_JSON_PATH`, default `./locations/patrol_waypoints.json`), (3) hardcoded defaults (differ under `use_sim`).

**Response:**
```json
{
  "status": "success",
  "location": {
    "predock": { "position": { "x": 0.5, "y": 0.0, "z": 0.0 }, "orientation": { "x": 0.0, "y": 0.0, "z": -0.7071, "w": 0.7071 } },
    "charger": { "position": { "x": 0.5, "y": -1.0, "z": 0.0 }, "orientation": { "x": 0.0, "y": 0.0, "z": -0.7071, "w": 0.7071 } }
  },
  "source": "map"
}
```

`source` is `"map"`, `"global"`, or `"default"` — which of the three locations the waypoints came from.

**Error conditions:** `400` if robot type is not Go2.

#### POST /charging/location/save

Save a per-map charging-station override (written atomically to `maps/<map_name>/<map_name>.charging_station.json`). Subsequent `GET /charging/location` calls for that map prefer it over the global file/defaults.

**Request body:**
```json
{
  "map_name": "kitchen",
  "predock": { "position": { "x": 0.5, "y": -2.0, "z": 0.0 }, "orientation": { "x": 0.0, "y": 0.0, "z": -0.7071, "w": 0.7071 } },
  "charger": { "position": { "x": 0.5, "y": -1.0, "z": 0.0 }, "orientation": { "x": 0.0, "y": 0.0, "z": -0.7071, "w": 0.7071 } }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `map_name` | string | Existing map folder under `maps/` to save the override for |
| `predock` | pose | Staging pose before the final docking approach |
| `charger` | pose | Pose at the charging station |

**Error conditions:** `400` if robot type is not Go2; `400` if `map_name`/`predock`/`charger` missing; `400` if pose validation fails; `400` if `map_name` is invalid (contains `/`, `\`, `..`, or a space) or its folder does not exist.

---

### Patrol (Go2 Only)

Autonomous patrol between waypoints along a route graph. If autocharging is enabled and the battery drops below a threshold, the robot docks, charges, and resumes patrol. Only available on the Unitree Go2.

> **Endpoint paths use the `<verb>/patrol` form** (`/start/patrol`, `/stop/patrol`, `/pause/patrol`, `/resume/patrol`).

#### POST /start/patrol

Start the autonomous patrol process using a route graph.

**Prerequisites:** Nav2 must be running.

**Request body:**
```json
{
  "launch_file": "go2_patrol_launch.py",
  "map_name": "office_floor1",
  "route_name": "patrol_route_1"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | **Yes** | Name of the map to patrol |
| `route_name` | string | **Yes** | Route graph file (GeoJSON) for patrol |
| `launch_file` | string | No | Custom launch file (default: `go2_patrol_launch.py`) |

**Response:**
```json
{ "status": "success", "message": "Patrol started" }
```

**Error conditions:** `400` if robot type is not Go2; `400` if Nav2 is not running; `400` if patrol is already running; `400` if `map_name` or `route_name` is missing; `400` if the route graph file does not exist; `500` if patrol failed to start.

#### POST /stop/patrol

Stop the patrol process.

**Response:**
```json
{ "status": "success", "message": "Patrol stopped" }
```

**Error conditions:** `400` if robot type is not Go2; `400` if patrol is not running.

#### POST /pause/patrol

Pause the currently running patrol.

**Response:**
```json
{ "status": "success", "message": "Patrol pause command sent" }
```

**Error conditions:** `400` if patrol is not running.

#### POST /resume/patrol

Resume the paused patrol.

**Response:**
```json
{ "status": "success", "message": "Patrol resume command sent" }
```

**Error conditions:** `400` if patrol is not running.

---

### Frontier Exploration (Go2 Only)

Autonomous frontier exploration runs as part of **2D SLAM**: starting `/start/slam/2d` launches the explorer, which begins exploring automatically — there is no separate start endpoint. Stopping SLAM tears it down. The endpoints below pause and resume it while SLAM stays up. (3D SLAM does not run Nav2 or the frontier explorer.)

#### POST /explore/stop

Pause frontier exploration (cancels the planning loop and any active goal).

**Response:**
```json
{ "status": "success", "message": "Frontier exploration stopped" }
```

#### POST /explore/resume

Resume frontier exploration (restarts the planning loop and triggers a new cycle).

**Response:**
```json
{ "status": "success", "message": "Frontier exploration resumed" }
```

#### GET /explore/status

Get the latest exploration status. The `message` field is a **JSON-encoded string**.

**Response:**
```json
{
  "status": "success",
  "message": "{\"running\": true, \"paused\": false, \"complete\": false, \"info\": \"Exploring\", \"status_received\": true}"
}
```

**Decoded `message` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `running` | `bool` | Whether exploration is active (SLAM running, not paused, not complete) |
| `paused` | `bool` | Whether the explorer was paused via `/explore/stop` |
| `complete` | `bool` | Whether exploration has completed |
| `info` | `str` | Additional status info |
| `status_received` | `bool` | Whether a status update has been received from the explorer yet |

---

### Map Management

#### POST /maps/save

Save the current SLAM map. This is the single map-save entry point: it writes **every** artifact the running SLAM stack can produce into one folder (there is no way to request only one artifact type). What lands in `maps/<map_name>/` depends on the SLAM mode:

| SLAM mode | Files written |
|-----------|---------------|
| 2D | `<name>.pgm`, `<name>.yaml`, `<name>.posegraph`, `<name>.data`, `<name>_vpr.npz` |
| 3D | the same, **plus** `<name>.pcd` (full resolution) and `<name>_downsampled.pcd` |

> If one artifact type fails while the other succeeds, the response is `200` with `"status": "partial_success"`, the written files in `files_created`, and failures in `errors`. **Check `status`, not just the HTTP code.**

**Request body:**
```json
{ "map_name": "office_floor1", "map_directory": "/custom/path/to/maps" }
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | **Yes** | Map name (no `/`, `\`, `..`, or spaces) |
| `map_directory` | string | No | Custom directory for map storage |

**Response:**
```json
{ "status": "success", "message": "Map saved successfully", "map_path": "/path/to/maps/office_floor1" }
```

**Error conditions:** `400` if SLAM is not running; `400` if `map_name` is missing or invalid.

#### GET /maps/list

List all saved maps.

**Response:**
```json
{ "maps": [ { "name": "office_floor1", "path": "/path/to/maps/office_floor1", "created": "2026-04-21T10:30:00" } ] }
```

#### POST /maps/delete

Delete a saved map.

**Request body:**
```json
{ "map_name": "office_floor1" }
```

**Response:**
```json
{ "status": "success", "message": "Map deleted successfully" }
```

**Error conditions:** `404` if the map does not exist.

---

### Route Management

#### POST /maps/route/save

Save a route graph (GeoJSON) to a map directory for use in navigation and patrol.

**Request body:**
```json
{
  "map_name": "office_floor1",
  "route_name": "patrol_route_1",
  "geojson": {
    "type": "FeatureCollection",
    "features": [
      { "type": "Feature", "geometry": { "type": "Point", "coordinates": [1.0, 2.0] }, "properties": { "id": "node_1", "type": "waypoint" } },
      { "type": "Feature", "geometry": { "type": "LineString", "coordinates": [[1.0, 2.0], [3.0, 4.0]] }, "properties": { "from": "node_1", "to": "node_2", "cost": 5.0 } }
    ]
  }
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | **Yes** | Map to associate with the route |
| `route_name` | string | **Yes** | Name for the route file |
| `geojson` | object | **Yes** | GeoJSON FeatureCollection of route nodes (Point) and edges (LineString) |

**Response:**
```json
{ "status": "success", "message": "Route 'patrol_route_1' saved" }
```

**Error conditions:** `400` if `map_name`/`route_name`/`geojson` missing; `404` if the map does not exist; `500` if writing the route file failed.

---

### Location Management

Named waypoints within maps, usable as navigation goals.

#### POST /maps/locations/add

Add a location with explicit pose data.

**Request body:**
```json
{
  "map_name": "office_floor1",
  "location": {
    "name": "reception",
    "description": "Front reception desk",
    "timestamp": "2026-04-21T10:30:00",
    "pose": {
      "position": {"x": 1.0, "y": 2.0, "z": 0.0},
      "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    }
  }
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | **Yes** | Map to add the location to |
| `location.name` | string | **Yes** | Unique name for this location |
| `location.description` | string | No | Human-readable description |
| `location.pose.position` | object | **Yes** | x, y, z in the map frame |
| `location.pose.orientation` | object | **Yes** | Quaternion (x, y, z, w) |

**Response:**
```json
{ "status": "success", "message": "Location added successfully" }
```

#### GET /maps/locations/list

List all saved locations across all maps (grouped by map name). The `message` field is a **JSON-encoded string**.

**Response:**
```json
{ "status": "success", "message": "{\"office_floor1\": [{\"name\": \"reception\", \"description\": \"Front desk\", ...}]}" }
```

#### POST /maps/locations/add/slam

Save the robot's current position as a named location during SLAM — the easiest way to add waypoints: drive the robot there and call this.

**Request body:**
```json
{ "map_name": "office_floor1", "label": "conference_room", "description": "Main conference room entrance" }
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | **Yes** | Map to add the location to |
| `label` | string | **Yes** | Unique name for this location |
| `description` | string | No | Human-readable description |

**Response:**
```json
{
  "status": "success",
  "message": "Location 'conference_room' saved successfully",
  "location": {
    "name": "conference_room",
    "description": "Main conference room entrance",
    "timestamp": "2026-04-21T10:30:00",
    "pose": {
      "position": {"x": 3.5, "y": 1.2, "z": 0.0},
      "orientation": {"x": 0.0, "y": 0.0, "z": 0.707, "w": 0.707}
    }
  }
}
```

**Error conditions:** `400` if SLAM is not running; `500` if unable to get the robot position from the map frame.

---

## Nav2 API (Port 5001)

Direct access to navigation control and real-time localization data.

### Quick Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Check API status |
| `GET` | `/api/pose` | Get current robot pose with covariance |
| `POST` | `/api/move_to_pose` | Send robot to a specific pose |
| `GET` | `/api/amcl_variance` | Get localization uncertainty |
| `GET` | `/api/nav2_status` | Get navigation goal status |
| `GET` | `/api/map` | Get current occupancy grid map |

---

### GET /api/status

Returns the API health status.

**Response:**
```json
{ "status": "OK", "message": "Go2 API is running" }
```

### GET /api/pose

Returns the current robot pose in the map frame with covariance.

**Response:**
```json
{
  "position": {"x": 0.0, "y": 0.0, "z": 0.0},
  "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
  "covariance": [0.0, ...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `position` | object | x, y, z in the map frame (meters) |
| `orientation` | object | Quaternion (x, y, z, w) |
| `covariance` | array | 36-element covariance matrix (6×6 flattened) |

### POST /api/move_to_pose

Send the robot to a pose in the map frame. Returns immediately — use `/api/nav2_status` to monitor progress.

**Request body:**
```json
{ "position": {"x": 1.0, "y": 2.0, "z": 0.0}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0} }
```

**Response:**
```json
{ "status": "success", "message": "Moving to specified pose" }
```

### GET /api/amcl_variance

Returns AMCL localization uncertainty (lower = more confident).

**Response:**
```json
{ "x_uncertainty": 0.1, "y_uncertainty": 0.1, "yaw_uncertainty": 5.0 }
```

| Field | Type | Description |
|-------|------|-------------|
| `x_uncertainty` | float | Position uncertainty in x (meters) |
| `y_uncertainty` | float | Position uncertainty in y (meters) |
| `yaw_uncertainty` | float | Heading uncertainty (degrees) |

### GET /api/nav2_status

Returns the status of active navigation goals.

**Response:**
```json
{ "nav2_status": [ { "goal_id": "abc123...", "status": "EXECUTING", "timestamp": {"sec": 1234567890, "nanosec": 123456789} } ] }
```

**Status values:** `UNKNOWN`, `ACCEPTED`, `EXECUTING`, `CANCELING`, `SUCCEEDED`, `CANCELED`, `ABORTED`.

### GET /api/map

Returns the current occupancy grid map.

**Response:**
```json
{
  "map_metadata": {
    "map_load_time": 1234567890.123456789,
    "resolution": 0.05,
    "width": 384,
    "height": 384,
    "origin": { "position": {"x": -10.0, "y": -10.0, "z": 0.0}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0} }
  },
  "data": [0, 0, 0, ...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `map_metadata.resolution` | float | Cell size (meters) |
| `map_metadata.width` / `height` | integer | Map size in cells |
| `map_metadata.origin` | object | Pose of the map origin (bottom-left corner) |
| `data` | array | Occupancy values: `-1` = unknown, `0` = free, `100` = occupied |
