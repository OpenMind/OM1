# QUICKSTART (5 minutes)
```bash
# Terminal 1: Start Bridge
cd tools/threejs-sim/bridge
npm install
node index.js

# Terminal 2: Start Client
cd tools/threejs-sim/client
npm install
npm run dev

# Open browser: http://localhost:5173
# Use W/A/S/D keys to control the robot
# Or click buttons in the HUD
```

## Expected Output

**Bridge (Terminal 1):**
```
bridge :8081
```

**Client (Terminal 2):**
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

## Test Commands
```bash
# Test action endpoint
curl -X POST http://localhost:8081/action \
  -H 'Content-Type: application/json' \
  -d '{"v": 0.05, "w": 0.0}'

# Reset simulation
curl -X POST http://localhost:8081/reset
```
