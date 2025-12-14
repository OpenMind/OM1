# Bounty #361 – OM1 × Three.js Mini Simulator

## Scope
Browser-based Three.js simulator with HTTP/WebSocket bridge for OM1 robot control.

## Changes
- `tools/threejs-sim/bridge/` - Express + WebSocket server
- `tools/threejs-sim/client/` - React + Three.js UI
- `docs/bounty-361/` - Complete documentation
- Sensor raycasting (13 beams, 4m range, 90° FOV)
- Collision detection with position rollback
- Reward function (forward +, collision -, rotation penalty)

## Features Implemented
✅ **Web Simulator**: Simple robot in Three.js that runs in browser
✅ **OM1 Control**: HTTP POST `/action` (move, rotate) and `/reset`
✅ **Environment**: Floor, walls, and 3 obstacles
✅ **Sensor System**: 13-beam raycasting for obstacle detection
✅ **HUD Display**: Real-time minDist, reward, steps, collisions
✅ **Controls**: Keyboard (WASD/Arrows) and button UI
✅ **Demo Video**: [Video link will be added]

## How to Test
See `docs/bounty-361/QUICKSTART.md` for full instructions.

**Quick test:**
```bash
# Terminal 1
cd tools/threejs-sim/bridge && npm install && node index.js

# Terminal 2
cd tools/threejs-sim/client && npm install && npm run dev
```

Open http://localhost:5173 and use W/A/S/D keys or click buttons.

## API Examples

**Move forward:**
```bash
curl -X POST http://localhost:8081/action \
  -H 'Content-Type: application/json' \
  -d '{"v": 0.05, "w": 0.0}'
```

**Turn left:**
```bash
curl -X POST http://localhost:8081/action \
  -H 'Content-Type: application/json' \
  -d '{"v": 0.0, "w": 0.12}'
```

**Reset:**
```bash
curl -X POST http://localhost:8081/reset
```

## Demo
Video: [YouTube/Drive link - will be added after recording]

## Architecture
```
OM1 Agent → HTTP POST → Bridge :8081 → WebSocket → Client :5173 (Three.js)
          ← WebSocket ←          ← State sync ←
```

## Notes
- Bridge runs on port 8081 (configurable via PORT env var)
- Client runs on port 5173 (Vite default)
- WebSocket broadcasts state every ~50ms
- Collision detection prevents robot from passing through obstacles
- Episode ends after 5 collisions or 1000 steps
