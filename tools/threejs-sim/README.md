# OM1 × Three.js Mini Simulator (PoC)

A minimal Three.js simulator with an HTTP/WebSocket bridge compatible with an OM1-style control loop.

## Features

- **Web Simulator**: 3D robot with obstacles, collision detection, and sensor beams
- **OM1 Control**: HTTP POST for actions, WebSocket for real-time state
- **Environment**: Simple playground with floor, walls, and obstacles
- **HUD**: Real-time display of sensors, reward, collisions, steps

## Quick Start

### Prerequisites
- Node.js 18+ and npm

### 1. Bridge (Backend Server)
```bash
cd bridge
npm install
npm start  # Runs on port 8081
```

### 2. Client (Frontend UI)
```bash
cd client
npm install
npm run dev  # Runs on http://localhost:5173
```

### 3. Open Browser
Navigate to http://localhost:5173

**Controls:**
- **W** or **↑**: Move forward
- **S** or **↓**: Move backward
- **A** or **←**: Turn left
- **D** or **→**: Turn right
- **R**: Reset simulation

## Bridge API (Node/Express + WS)

### POST /action
Apply motion command and get response:
```bash
curl -X POST http://localhost:8081/action \
  -H 'Content-Type: application/json' \
  -d '{"v": 0.05, "w": 0.0}'
```

**Request body:**
- `v` (float): Linear velocity (forward/backward)
- `w` (float): Angular velocity (rotation)

**Response:**
```json
{
  "ok": true,
  "sensors": {
    "maxRange": 4,
    "fov": 1.5708,
    "beams": 13,
    "distances": [4.0, 3.8, 3.5, ...]
  },
  "reward": 0.045,
  "done": false,
  "info": {
    "steps": 42,
    "collisions": 0,
    "minDist": 1.2
  }
}
```

### POST /reset
Reset the episode:
```bash
curl -X POST http://localhost:8081/reset
```

**Response:**
```json
{
  "ok": true
}
```

### WebSocket /ws
Real-time state broadcasts every ~50ms:
```javascript
const ws = new WebSocket('ws://localhost:8081/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
  // {
  //   type: 'state',
  //   pose: { x: 0, z: 0, yaw: 0 },
  //   sensors: { maxRange: 4, fov: 1.57, beams: 13, distances: [...] },
  //   reward: 0.045,
  //   done: false
  // }
};
```

## Client (React + Three.js)

Built with:
- React 19
- Three.js + React Three Fiber
- @react-three/drei for helpers
- Vite for dev server

The client visualizes:
- 3D robot (box geometry)
- Obstacles (3 boxes)
- Sensor beams (colored by distance)
- Floor grid
- Real-time HUD

## Environment Configuration

**Bridge:**
```bash
PORT=8081 node index.js
```

**Client:**
```bash
VITE_BRIDGE_WS=ws://192.168.1.100:8081/ws npm run dev
```

## Example Python Agent
```python
import requests
import time

BRIDGE_URL = "http://localhost:8081"

def main():
    # Reset
    requests.post(f"{BRIDGE_URL}/reset")

    for i in range(100):
        # Move forward
        resp = requests.post(
            f"{BRIDGE_URL}/action",
            json={"v": 0.05, "w": 0.0}
        ).json()

        print(f"Step {i}: reward={resp['reward']:.3f}, "
              f"minDist={resp['info']['minDist']:.2f}")

        if resp['done']:
            print("Episode done!")
            break

        time.sleep(0.1)

if __name__ == "__main__":
    main()
```

## Architecture
```
┌─────────────┐     HTTP POST      ┌──────────┐
│   OM1 AI    │ ──────────────────> │  Bridge  │
│   Agent     │ <────────────────── │  :8081   │
└─────────────┘   WebSocket state   └──────────┘
                                         │
                                         │ State sync
                                         ▼
                                    ┌──────────┐
                                    │  Client  │
                                    │  :5173   │
                                    │ Three.js │
                                    └──────────┘
```

## Reward Function
```
reward = forward_bonus - backward_penalty - rotation_penalty - collision_penalty
       = (v >= 0 ? +|v| : -0.5*|v|) - 0.05*|w| - (collided ? 1.0 : 0)
```

## Episode Termination

Episode ends when:
- `collisions >= 5`
- `steps >= 1000`

## Development

### Project Structure
```
tools/threejs-sim/
├── bridge/           # Express + WebSocket server
│   ├── index.js
│   └── package.json
├── client/           # React + Three.js UI
│   ├── src/
│   │   ├── App.jsx
│   │   └── ...
│   └── package.json
└── README.md
```

### Sensor Configuration
- **Beams**: 13 rays
- **FOV**: 90° (π/2 radians)
- **Max Range**: 4 meters
- **Ray Color**: Blue (far) → Red (near)

### Collision Detection
- AABB 2D raycasting against obstacles
- Position rollback on collision
- Collision counter incremented

## License
This project is licensed under the MIT License. See the main OM1 LICENSE file for details.
