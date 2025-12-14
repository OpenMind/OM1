# Bounty #361 — OM1 × Three.js Mini Simulator

A browser-based Three.js robot simulator with HTTP/WebSocket bridge compatible with OM1 control loops.

## Features

- **Web Simulator**: 3D robot with obstacles, collision detection, and sensor beams
- **OM1 Control**: HTTP POST for actions, WebSocket for real-time state
- **Environment**: Simple playground with floor, walls, and obstacles
- **HUD**: Real-time display of sensors, reward, collisions, steps

## Quick Start (5 minutes)

### Prerequisites
- Node.js 18+ and npm

### 1. Bridge (Backend)
```bash
cd tools/threejs-sim/bridge
npm install
node index.js  # Runs on port 8081
```

### 2. Client (Frontend)
```bash
cd tools/threejs-sim/client
npm install
npm run dev  # Runs on http://localhost:5173
```

### 3. Test
Open browser: http://localhost:5173

**Controls:**
- W/↑: Forward
- S/↓: Backward
- A/←: Turn left
- D/→: Turn right
- R: Reset

## API Integration with OM1

### POST /action
```bash
curl -X POST http://localhost:8081/action \
  -H 'Content-Type: application/json' \
  -d '{"v": 0.05, "w": 0.0}'

# Response:
{
  "ok": true,
  "sensors": {
    "maxRange": 4,
    "fov": 1.5708,
    "beams": 13,
    "distances": [4.0, 3.8, ...]
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
```bash
curl -X POST http://localhost:8081/reset
```

### WebSocket /ws
```javascript
const ws = new WebSocket('ws://localhost:8081/ws');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.type === 'state'
  // data.pose, data.sensors, data.reward, data.done
};
```

## Demo
Video: [Coming soon - will be added after recording]

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

## Development

### Environment Variables
Bridge supports `PORT` env var (default: 8081)

Client supports `VITE_BRIDGE_WS` for custom bridge URL:
```bash
VITE_BRIDGE_WS=ws://192.168.1.100:8081/ws npm run dev
```

### Testing
```bash
# Manual keyboard control
npm run dev

# Python script example (see example_agent.py)
python tools/threejs-sim/example_agent.py
```
