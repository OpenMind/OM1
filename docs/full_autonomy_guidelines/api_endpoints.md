---
title: Premium Features API Endpoints
description: "REST API documentation for OM1 ROS2 SDK"
icon: link
---

## Overview

The OM1 ROS2 SDK provides two REST APIs for remote control and monitoring of the robot:

| API | Port | Description |
|-----|------|-------------|
| **Orchestrator API** | `5000` | System orchestration and high-level control (SLAM, Nav2, charging, patrol, maps) |
| **Nav2 API** | `5001` | Direct navigation control and real-time monitoring (pose, goals, localization) |

> **Note:** These APIs are part of the **Premium Features** available through the Enterprise Plan. See [Premium Features](../developing/premium_features.md) for details.

---

## Orchestrator API (Port 5000)

The Orchestrator API manages the robot's operational state, including SLAM, navigation, charging, patrol, and map management. All endpoints return JSON responses.

### Quick Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/status` | Get status of all running processes |
| `POST` | `/start/base_control` | Start base control process |
| `POST` | `/stop/base_control` | Stop base control process |
| `POST` | `/start/slam` | Start SLAM mapping |
| `POST` | `/stop/slam` | Stop SLAM mapping |
| `POST` | `/start/nav2` | Start Nav2 navigation stack |
| `POST` | `/stop/nav2` | Stop Nav2 navigation stack |
| `POST` | `/charging/dock` | Start autonomous docking (Go2 only) |
| `POST` | `/charging/stop` | Stop docking process |
| `GET` | `/charging/status` | Get charging status |
| `POST` | `/patrol/start` | Start patrol (Go2 only) |
| `POST` | `/patrol/stop` | Stop patrol |
| `POST` | `/patrol/pause` | Pause patrol |
| `POST` | `/patrol/resume` | Resume patrol |
| `POST` | `/maps/save` | Save current SLAM map |
| `GET` | `/maps/list` | List all saved maps |
| `POST` | `/maps/delete` | Delete a saved map |
| `POST` | `/maps/locations/add` | Add a location to a map |
| `GET` | `/maps/locations/list` | List all saved locations |
| `POST` | `/maps/locations/add/slam` | Save current position as location |

---

### System Status

#### GET /status

Returns the operational status of all robot processes. Use this to check which services are currently active before starting or stopping processes.

**Response:**
```json
{
  "status": "success",
  "message": "{\"base_control_running\": true, \"slam_running\": false, \"nav2_running\": true, \"patrol_running\": false}"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `base_control_running` | boolean | Whether base motor control is active |
| `slam_running` | boolean | Whether SLAM mapping is in progress |
| `nav2_running` | boolean | Whether Nav2 navigation stack is active |
| `patrol_running` | boolean | Whether autonomous patrol is running |

---

### Base Control

Base control manages the low-level motor commands that allow the robot to move. It must be running for any movement operations.

#### POST /start/base_control

Start the base control process. This initializes the robot's motor controller and enables movement commands.

**Request body (optional):**
```json
{
  "launch_file": "base_control_launch.py"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `launch_file` | string | No | Custom launch file (default: `base_control_launch.py`) |

**Response:**
```json
{
  "status": "success",
  "message": "Base control started"
}
```

**Error conditions:**

| Status | Condition |
|--------|-----------|
| `400` | SLAM or Nav2 is already running |

#### POST /stop/base_control

Stop the base control process. This disables all motor commands.

**Response:**
```json
{
  "status": "success",
  "message": "Base control stopped"
}
```

---

### SLAM

SLAM (Simultaneous Localization and Mapping) allows the robot to build a map of its environment while tracking its position within that map.

#### POST /start/slam

Start the SLAM process for mapping. The robot will begin building an occupancy grid map as it moves through the environment.

**Request body (optional):**
```json
{
  "launch_file": "slam_launch.py",
  "map_yaml": "/path/to/existing/map.yaml"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `launch_file` | string | No | Custom SLAM launch file |
| `map_yaml` | string | No | Path to existing map to continue mapping from |

**Response:**
```json
{
  "status": "success",
  "message": "SLAM started"
}
```

**Error conditions:**

| Status | Condition |
|--------|-----------|
| `400` | Nav2 is running (must stop Nav2 before starting SLAM) |

#### POST /stop/slam

Stop the SLAM process. Remember to save the map using `/maps/save` before stopping if you want to keep the map.

**Response:**
```json
{
  "status": "success",
  "message": "SLAM stopped"
}
```

---

### Navigation (Nav2)

Nav2 is the ROS2 navigation stack that enables autonomous navigation using a pre-built map. It handles path planning, obstacle avoidance, and localization.

#### POST /start/nav2

Start the Nav2 navigation stack with a saved map. The robot will localize itself within the map and be ready to receive navigation goals.

**Request body:**
```json
{
  "map_name": "office_floor1",
  "launch_file": "nav2_launch.py"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | **Yes** | Name of the saved map to use for navigation |
| `launch_file` | string | No | Custom Nav2 launch file |

**Response:**
```json
{
  "status": "success",
  "message": "Nav2 started"
}
```

**Error conditions:**

| Status | Condition |
|--------|-----------|
| `400` | SLAM is running (must stop SLAM before starting Nav2) |
| `400` | `map_name` is not provided |

#### POST /stop/nav2

Stop the Nav2 navigation stack. This will cancel any active navigation goals.

**Response:**
```json
{
  "status": "success",
  "message": "Nav2 stopped"
}
```

---

### Charging (Go2 Only)

The charging endpoints control the autonomous docking and charging process. The robot uses AprilTag visual servoing to precisely align with the charging dock.

> **Note:** These endpoints are only available on the Unitree Go2 robot.

#### POST /charging/dock

Start the autonomous docking sequence. The robot will navigate to the charging station and align itself with the dock contacts.

**Prerequisites:**
- Nav2 must be running
- Robot must not already be charging

**Request body (optional):**
```json
{
  "launch_file": "go2_charge_launch.py"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `launch_file` | string | No | Custom charging launch file |

**Response:**
```json
{
  "status": "success",
  "message": "Charging dock process started"
}
```

**Error conditions:**

| Status | Condition |
|--------|-----------|
| `400` | Robot type is not Go2 |
| `400` | Nav2 is not running |
| `400` | Already charging or dock process is running |

#### POST /charging/stop

Stop the charging dock process. Use this to abort docking or undock the robot.

**Response:**
```json
{
  "status": "success",
  "message": "Charging dock process stopped"
}
```

#### GET /charging/status

Get the current charging and docking status. Use this to monitor battery level and charging state.

**Response:**
```json
{
  "is_charging": false,
  "battery_percentage": 85,
  "dock_process_running": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `is_charging` | boolean | Whether the robot is currently charging |
| `battery_percentage` | integer | Current battery level (0-100) |
| `dock_process_running` | boolean | Whether the docking sequence is in progress |

---

### Patrol (Go2 Only)

The patrol endpoints control autonomous patrol behavior. The robot will navigate between predefined waypoints, monitoring for activity. If autocharging is enabled, and battery drops below a particular level, the robot will go to the docking station, charge and continue patrol.

> **Note:** These endpoints are only available on the Unitree Go2 robot.

#### POST /patrol/start

Start the autonomous patrol process. The robot will begin navigating between saved patrol waypoints.

**Prerequisites:**
- Nav2 must be running
- Patrol waypoints must be configured

**Request body (optional):**
```json
{
  "launch_file": "go2_patrol_launch.py"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `launch_file` | string | No | Patrol launch file |

**Response:**
```json
{
  "status": "success",
  "message": "Patrol started"
}
```

**Error conditions:**

| Status | Condition |
|--------|-----------|
| `400` | Robot type is not Go2 |
| `400` | Nav2 is not running |
| `400` | Patrol is already running |

#### POST /patrol/stop

Stop the patrol process. The robot will stop at its current position.

**Response:**
```json
{
  "status": "success",
  "message": "Patrol stopped"
}
```

#### POST /patrol/pause

Pause the currently running patrol. The robot will hold its position until resumed.

**Response:**
```json
{
  "status": "success",
  "message": "Patrol pause command sent"
}
```

#### POST /patrol/resume

Resume the paused patrol from where it left off.

**Response:**
```json
{
  "status": "success",
  "message": "Patrol resume command sent"
}
```

---

### Map Management

Map management endpoints allow you to save, list, and delete SLAM-generated maps. Maps are stored on the robot and can be loaded later for navigation.

#### POST /maps/save

Save the current SLAM map to a file. Call this before stopping SLAM to preserve the map.

**Request body:**
```json
{
  "map_name": "office_floor1",
  "map_directory": "/custom/path/to/maps"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | **Yes** | Name for the saved map (no `/`, `\`, `..`, or spaces) |
| `map_directory` | string | No | Custom directory path for map storage |

**Response:**
```json
{
  "status": "success",
  "message": "Map saved successfully",
  "map_path": "/path/to/maps/office_floor1"
}
```

**Error conditions:**

| Status | Condition |
|--------|-----------|
| `400` | SLAM is not running |
| `400` | `map_name` is missing or contains invalid characters |

#### GET /maps/list

List all saved maps with their metadata.

**Response:**
```json
{
  "maps": [
    {
      "name": "office_floor1",
      "path": "/path/to/maps/office_floor1",
      "created": "2026-04-21T10:30:00"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Map identifier |
| `path` | string | Full path to map files |
| `created` | string | ISO 8601 timestamp of map creation |

#### POST /maps/delete

Delete a saved map.

**Request body:**
```json
{
  "map_name": "office_floor1"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | **Yes** | Name of the map to delete |

**Response:**
```json
{
  "status": "success",
  "message": "Map deleted successfully"
}
```

**Error conditions:**

| Status | Condition |
|--------|-----------|
| `404` | Map does not exist |

---

### Location Management

Location management endpoints allow you to save and manage named waypoints within maps. These locations can be used as navigation goals.

#### POST /maps/locations/add

Add a named location to a map's location list with explicit pose data.

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
| `map_name` | string | **Yes** | Name of the map to add the location to |
| `location.name` | string | **Yes** | Unique name for this location |
| `location.description` | string | No | Human-readable description |
| `location.pose.position` | object | **Yes** | x, y, z coordinates in map frame |
| `location.pose.orientation` | object | **Yes** | Quaternion (x, y, z, w) orientation |

**Response:**
```json
{
  "status": "success",
  "message": "Location added successfully"
}
```

#### GET /maps/locations/list

List all saved locations across all maps. Returns locations grouped by map name.

**Response:**
```json
{
  "status": "success",
  "message": "{\"office_floor1\": [{\"name\": \"reception\", \"description\": \"Front desk\", ...}]}"
}
```

#### POST /maps/locations/add/slam

Save the robot's current position as a named location during SLAM. This is the easiest way to add waypoints — just drive the robot to a location and call this endpoint.

**Request body:**
```json
{
  "map_name": "office_floor1",
  "label": "conference_room",
  "description": "Main conference room entrance"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | **Yes** | Name of the map to add the location to |
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

**Error conditions:**

| Status | Condition |
|--------|-----------|
| `400` | SLAM is not running |
| `500` | Unable to get robot position from map frame |

---

## Nav2 API (Port 5001)

The Nav2 API provides direct access to navigation control and real-time localization data. Use this API for sending navigation goals and monitoring the robot's position and navigation status.

### Quick Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Check API status |
| `GET` | `/api/pose` | Get current robot pose |
| `POST` | `/api/move_to_pose` | Send robot to a specific pose |
| `GET` | `/api/amcl_variance` | Get localization uncertainty |
| `GET` | `/api/nav2_status` | Get navigation goal status |
| `GET` | `/api/map` | Get current occupancy grid map |

---

### GET /api/status

Returns the API health status. Use this to verify the Nav2 API is running and accessible.

**Response:**
```json
{
  "status": "OK",
  "message": "Go2 API is running"
}
```

---

### GET /api/pose

Returns the current robot pose in the map frame with covariance matrix. The covariance indicates localization uncertainty.

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
| `position` | object | x, y, z coordinates in map frame (meters) |
| `orientation` | object | Quaternion (x, y, z, w) representing heading |
| `covariance` | array | 36-element covariance matrix (6x6 flattened) |

---

### POST /api/move_to_pose

Send the robot to a specific pose in the map frame. This creates a navigation goal and returns immediately — use `/api/nav2_status` to monitor progress.

**Request body:**
```json
{
  "position": {"x": 1.0, "y": 2.0, "z": 0.0},
  "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `position` | object | **Yes** | Target x, y, z coordinates in map frame |
| `orientation` | object | **Yes** | Target orientation as quaternion |

**Response:**
```json
{
  "status": "success",
  "message": "Moving to specified pose"
}
```

---

### GET /api/amcl_variance

Returns the AMCL localization uncertainty estimates. Lower values indicate more confident localization.

**Response:**
```json
{
  "x_uncertainty": 0.1,
  "y_uncertainty": 0.1,
  "yaw_uncertainty": 5.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `x_uncertainty` | float | Position uncertainty in x (meters) |
| `y_uncertainty` | float | Position uncertainty in y (meters) |
| `yaw_uncertainty` | float | Heading uncertainty (degrees) |

---

### GET /api/nav2_status

Returns the status of all active navigation goals. Use this to monitor navigation progress after calling `/api/move_to_pose`.

**Response:**
```json
{
  "nav2_status": [
    {
      "goal_id": "abc123...",
      "status": "EXECUTING",
      "timestamp": {"sec": 1234567890, "nanosec": 123456789}
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `goal_id` | string | Unique identifier for the navigation goal |
| `status` | string | Current status of the goal |
| `timestamp` | object | Time when status was recorded |

**Status Values:**

| Status | Description |
|--------|-------------|
| `UNKNOWN` | Goal status is unknown |
| `ACCEPTED` | Goal has been accepted but not yet started |
| `EXECUTING` | Robot is actively navigating to goal |
| `CANCELING` | Goal cancellation in progress |
| `SUCCEEDED` | Robot reached the goal successfully |
| `CANCELED` | Goal was canceled |
| `ABORTED` | Navigation failed (obstacle, timeout, etc.) |

---

### GET /api/map

Returns the current occupancy grid map. The map is represented as a 2D array where each cell indicates occupancy probability.

**Response:**
```json
{
  "map_metadata": {
    "map_load_time": 1234567890.123456789,
    "resolution": 0.05,
    "width": 384,
    "height": 384,
    "origin": {
      "position": {"x": -10.0, "y": -10.0, "z": 0.0},
      "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    }
  },
  "data": [0, 0, 0, ...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `map_metadata.resolution` | float | Size of each cell in meters |
| `map_metadata.width` | integer | Map width in cells |
| `map_metadata.height` | integer | Map height in cells |
| `map_metadata.origin` | object | Pose of the map origin (bottom-left corner) |
| `data` | array | Occupancy values: `-1` = unknown, `0` = free, `100` = occupied |

---
