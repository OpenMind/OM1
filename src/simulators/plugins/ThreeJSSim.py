import asyncio
import logging
import math
import os
import threading
import time
from typing import List, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from llm.output_model import Action
from simulators.base import Simulator, SimulatorConfig


class ThreeJSSim(Simulator):
    """Three.js browser-based robot simulator."""

    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        self.messages: List[str] = []
        self._initialized = False
        self._lock = threading.Lock()
        self.active_connections: Set[WebSocket] = set()

        self.robot_state = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "moving": False,
            "current_action": "idle",
        }

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.static_dir = os.path.join(self.base_dir, "threejs_assets")
        os.makedirs(self.static_dir, exist_ok=True)

        logging.info("Initializing ThreeJSSim...")

        self.app = FastAPI()
        self.app.mount("/static", StaticFiles(directory=self.static_dir), name="static")

        @self.app.get("/")
        async def get_index():
            return await self._get_simulator_html()

        @self.app.post("/api/command")
        async def receive_command(command: dict):
            try:
                with self._lock:
                    action_type = command.get("type") or command.get("action", "")

                    if action_type == "turn_left" or action_type == "turn left":
                        target_yaw = command.get("target_yaw")
                        if target_yaw is not None:
                            self.robot_state["yaw"] = target_yaw
                        else:
                            self.robot_state["yaw"] = self._normalize_angle(
                                self.robot_state["yaw"] - 90
                            )
                        self.robot_state["current_action"] = "turn left"
                    elif action_type == "turn_right" or action_type == "turn right":
                        target_yaw = command.get("target_yaw")
                        if target_yaw is not None:
                            self.robot_state["yaw"] = target_yaw
                        else:
                            self.robot_state["yaw"] = self._normalize_angle(
                                self.robot_state["yaw"] + 90
                            )
                        self.robot_state["current_action"] = "turn right"
                    elif (
                        action_type == "move_forward" or action_type == "move forwards"
                    ):
                        self.robot_state["current_action"] = "move forwards"
                        self.robot_state["moving"] = True
                        yaw_rad = math.radians(self.robot_state["yaw"])
                        self.robot_state["x"] -= math.sin(yaw_rad) * 0.5
                        self.robot_state["z"] -= math.cos(yaw_rad) * 0.5
                        logging.info(
                            f"Robot moving forwards to ({self.robot_state['x']:.2f}, {self.robot_state['z']:.2f})"
                        )
                    elif action_type == "move_back" or action_type == "move back":
                        self.robot_state["current_action"] = "move back"
                        self.robot_state["moving"] = True
                        yaw_rad = math.radians(self.robot_state["yaw"])
                        self.robot_state["x"] += math.sin(yaw_rad) * 0.5
                        self.robot_state["z"] += math.cos(yaw_rad) * 0.5
                        logging.info(
                            f"Robot moving back to ({self.robot_state['x']:.2f}, {self.robot_state['z']:.2f})"
                        )
                    elif action_type == "stop" or action_type == "stand still":
                        self.robot_state["current_action"] = "stand still"
                        self.robot_state["moving"] = False
                    elif action_type == "move":
                        self.robot_state["moving"] = True
                    elif action_type == "rotate":
                        self.robot_state["moving"] = True

                # Broadcast updated state
                await self.broadcast_state()
                return {"status": "ok", "robot_state": self.robot_state}
            except Exception as e:
                logging.error(f"Error processing command: {e}")
                return {"status": "error", "message": str(e)}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.add(websocket)
            try:
                await websocket.send_json(self.robot_state)
                while True:
                    try:
                        data = await websocket.receive_text()
                        logging.debug(f"Received from client: {data}")
                    except WebSocketDisconnect:
                        break
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
            finally:
                self.active_connections.discard(websocket)

        try:
            logging.info("Starting ThreeJSSim server thread...")
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            time.sleep(1)

            if self.server_thread.is_alive():
                port = self.config.port or 8001
                logging.info(
                    f"\033[1;36mThreeJSSim server started successfully - Open http://localhost:{port} in your browser\033[0m"
                )
                self._initialized = True
            else:
                logging.error("ThreeJSSim server failed to start")
        except Exception as e:
            logging.error(f"Error starting ThreeJSSim server thread: {e}")

    def _run_server(self):
        """Run the FastAPI server."""
        port = self.config.port or 8001
        host = self.config.host or "0.0.0.0"
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="error",
            server_header=False,
        )
        server = uvicorn.Server(config)
        server.run()

    async def _get_simulator_html(self) -> HTMLResponse:
        """Generate the HTML for the three.js simulator."""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OM1 Three.js Simulator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            overflow: hidden;
            background: #1a1a1a;
        }
        #canvas-container {
            width: 100vw;
            height: 100vh;
            position: relative;
        }
        #info-panel {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-size: 14px;
            z-index: 100;
            min-width: 250px;
        }
        #info-panel h3 {
            margin-bottom: 10px;
            color: #4CAF50;
        }
        #info-panel div {
            margin: 5px 0;
        }
        #status {
            color: #4CAF50;
            font-weight: bold;
        }
        .disconnected {
            color: #f44336;
        }
        #controls {
            position: absolute;
            bottom: 20px;
            right: 20px;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            z-index: 100;
            width: 120px;
        }
        .control-btn {
            width: 36px;
            height: 36px;
            padding: 0;
            font-size: 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            background: rgba(33, 150, 243, 0.9);
            color: white;
            transition: all 0.2s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            backdrop-filter: blur(10px);
        }
        .control-btn:hover {
            background: rgba(25, 118, 210, 0.95);
            transform: scale(1.1);
            box-shadow: 0 4px 8px rgba(0,0,0,0.4);
        }
        .control-btn:active {
            transform: scale(0.95);
        }
        .control-btn.stop {
            grid-column: 2;
            background: rgba(244, 67, 54, 0.9);
        }
        .control-btn.stop:hover {
            background: rgba(211, 47, 47, 0.95);
        }
    </style>
</head>
<body>
    <div id="canvas-container"></div>
    <div id="info-panel">
        <h3>OM1 Three.js Simulator</h3>
        <div>Status: <span id="status">Connecting...</span></div>
        <div>Position: <span id="position">0, 0</span></div>
        <div>Rotation: <span id="rotation">0°</span></div>
        <div>Action: <span id="action">idle</span></div>
    </div>
    <div id="controls">
        <button class="control-btn" onclick="sendCommand('turn left')" title="Turn Left">↶</button>
        <button class="control-btn" onclick="sendCommand('move forwards')" title="Move Forward">↑</button>
        <button class="control-btn" onclick="sendCommand('turn right')" title="Turn Right">↷</button>
        <button class="control-btn" onclick="sendCommand('move back')" title="Move Back">↓</button>
        <button class="control-btn stop" onclick="sendCommand('stand still')" title="Stop">⏸</button>
    </div>

    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
    }
    </script>

    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x87CEEB);
        scene.fog = new THREE.Fog(0x87CEEB, 10, 50);
        const camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        camera.position.set(5, 5, 5);
        camera.lookAt(0, 0, 0);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        document.getElementById('canvas-container').appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0, 0, 0);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.camera.left = -10;
        directionalLight.shadow.camera.right = 10;
        directionalLight.shadow.camera.top = 10;
        directionalLight.shadow.camera.bottom = -10;
        scene.add(directionalLight);

        const groundGeometry = new THREE.PlaneGeometry(20, 20);
        const groundMaterial = new THREE.MeshStandardMaterial({ 
            color: 0x90EE90,
            roughness: 0.8,
            metalness: 0.2
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        scene.add(ground);

        const gridHelper = new THREE.GridHelper(20, 20, 0x888888, 0x444444);
        scene.add(gridHelper);

        const wallMaterial = new THREE.MeshStandardMaterial({ 
            color: 0x888888,
            roughness: 0.7
        });

        const walls = [
            { pos: [0, 1, -8], size: [20, 2, 0.5] },
            { pos: [0, 1, 8], size: [20, 2, 0.5] },
            { pos: [-8, 1, 0], size: [0.5, 2, 20] },
            { pos: [8, 1, 0], size: [0.5, 2, 20] },
        ];

        walls.forEach(wall => {
            const geometry = new THREE.BoxGeometry(...wall.size);
            const mesh = new THREE.Mesh(geometry, wallMaterial);
            mesh.position.set(...wall.pos);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            scene.add(mesh);
        });

        const robotGroup = new THREE.Group();
        
        const bodyGeometry = new THREE.BoxGeometry(0.6, 0.4, 0.8);
        const bodyMaterial = new THREE.MeshStandardMaterial({ 
            color: 0x2196F3,
            roughness: 0.5,
            metalness: 0.3
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 0.2;
        body.castShadow = true;
        robotGroup.add(body);

        const headGeometry = new THREE.BoxGeometry(0.3, 0.3, 0.3);
        const headMaterial = new THREE.MeshStandardMaterial({ 
            color: 0xFF9800,
            roughness: 0.5
        });
        const head = new THREE.Mesh(headGeometry, headMaterial);
        head.position.set(0, 0.5, 0.2);
        head.castShadow = true;
        robotGroup.add(head);

        const wheelGeometry = new THREE.CylinderGeometry(0.15, 0.15, 0.1, 16);
        const wheelMaterial = new THREE.MeshStandardMaterial({ 
            color: 0x333333,
            roughness: 0.9
        });

        const wheelPositions = [
            [0.3, 0.1, 0.35],
            [-0.3, 0.1, 0.35],
            [0.3, 0.1, -0.35],
            [-0.3, 0.1, -0.35],
        ];

        wheelPositions.forEach(pos => {
            const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
            wheel.rotation.z = Math.PI / 2;
            wheel.position.set(...pos);
            wheel.castShadow = true;
            robotGroup.add(wheel);
        });

        robotGroup.position.set(0, 0, 0);
        scene.add(robotGroup);

        let robotState = {
            x: 0,
            z: 0,
            yaw: 0,
            moving: false,
            current_action: 'idle'
        };

        let targetYaw = 0;
        let targetX = 0;
        let targetZ = 0;
        let isRotating = false;
        let isMoving = false;
        const moveSpeed = 0.02;
        const rotateSpeed = 2;
        let mockWs = null;
        const mockWsUrl = 'ws://localhost:8765';

        function connectMockWebSocket() {
            try {
                mockWs = new WebSocket(mockWsUrl);
                mockWs.onopen = () => {
                    console.log('Connected to MockInput WebSocket');
                };
                mockWs.onerror = (error) => {
                    console.debug('MockInput WebSocket not available');
                };
                mockWs.onclose = () => {};
            } catch (e) {
                console.debug('Could not connect to MockInput WebSocket:', e);
            }
        }

        window.sendCommand = function(command) {
            console.log('Sending command:', command);
            
            handleAction(command);
            
            fetch('/api/command', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    action: command
                })
            }).then(response => {
                if (response.ok) {
                    console.log('Command sent via API:', command);
                    return response.json();
                } else {
                    console.error('API error:', response.status);
                }
            }).then(data => {
                if (data && data.robot_state) {
                    robotState = { ...robotState, ...data.robot_state };
                    updateUI();
                }
            }).catch(error => {
                console.error('Error sending command:', error);
            });
            
            if (mockWs && mockWs.readyState === WebSocket.OPEN) {
                mockWs.send(command);
                console.log('Command also sent via MockInput WebSocket');
            }
        };
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 10;

        function connectWebSocket() {
            try {
                ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    console.log('Connected to WebSocket');
                    document.getElementById('status').textContent = 'Connected';
                    document.getElementById('status').className = '';
                    reconnectAttempts = 0;
                };

                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (!isMoving && !isRotating) {
                            if (data.z !== undefined) {
                                robotState.z = data.z;
                            }
                            if (data.x !== undefined) {
                                robotState.x = data.x;
                            }
                            if (data.yaw !== undefined) {
                                robotState.yaw = data.yaw;
                            }
                            if (data.current_action !== undefined) {
                                robotState.current_action = data.current_action;
                            }
                        } else {
                            if (data.current_action !== undefined) {
                                robotState.current_action = data.current_action;
                            }
                        }
                        updateUI();
                        
                        if (data.current_action && !isMoving && !isRotating) {
                            handleAction(data.current_action);
                        }
                    } catch (e) {
                        console.error('Error parsing WebSocket message:', e);
                    }
                };

                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    document.getElementById('status').textContent = 'Error';
                    document.getElementById('status').className = 'disconnected';
                };

                ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    document.getElementById('status').textContent = 'Disconnected';
                    document.getElementById('status').className = 'disconnected';
                    
                    // Attempt to reconnect
                    if (reconnectAttempts < maxReconnectAttempts) {
                        reconnectAttempts++;
                        setTimeout(connectWebSocket, 2000);
                    }
                };
            } catch (e) {
                console.error('Error connecting WebSocket:', e);
            }
        }

        function normalizeAngle(angle) {
            while (angle < -180) angle += 360;
            while (angle > 180) angle -= 360;
            return angle;
        }

        function handleAction(action) {
            switch(action) {
                case 'turn left':
                    targetYaw = normalizeAngle(robotState.yaw - 90);
                    isRotating = true;
                    isMoving = false;
                    break;
                case 'turn right':
                    targetYaw = normalizeAngle(robotState.yaw + 90);
                    isRotating = true;
                    isMoving = false;
                    break;
                case 'move forwards':
                    const yawRad = THREE.MathUtils.degToRad(robotState.yaw);
                    targetX = robotState.x - Math.sin(yawRad) * 0.5;
                    targetZ = robotState.z - Math.cos(yawRad) * 0.5;
                    isMoving = true;
                    isRotating = false;
                    break;
                case 'move back':
                    const yawRadBack = THREE.MathUtils.degToRad(robotState.yaw);
                    targetX = robotState.x + Math.sin(yawRadBack) * 0.5;
                    targetZ = robotState.z + Math.cos(yawRadBack) * 0.5;
                    isMoving = true;
                    isRotating = false;
                    break;
                case 'stand still':
                case 'idle':
                    isMoving = false;
                    isRotating = false;
                    break;
            }
        }

        function updateUI() {
            document.getElementById('position').textContent = 
                `${robotState.x.toFixed(2)}, ${robotState.z.toFixed(2)}`;
            document.getElementById('rotation').textContent = `${robotState.yaw.toFixed(1)}°`;
            document.getElementById('action').textContent = robotState.current_action || 'idle';
        }

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);

            if (isRotating) {
                let angleDiff = targetYaw - robotState.yaw;
                if (angleDiff > 180) angleDiff -= 360;
                if (angleDiff < -180) angleDiff += 360;
                
                if (Math.abs(angleDiff) > 0.5) {
                    const rotateDir = angleDiff > 0 ? 1 : -1;
                    robotState.yaw += rotateDir * rotateSpeed;
                    robotState.yaw = normalizeAngle(robotState.yaw);
                } else {
                    robotState.yaw = targetYaw;
                    isRotating = false;
                }
            }
            if (isMoving) {
                const dx = targetX - robotState.x;
                const dz = targetZ - robotState.z;
                const distance = Math.sqrt(dx * dx + dz * dz);
                
                if (distance > 0.01) {
                    robotState.x += (dx / distance) * moveSpeed;
                    robotState.z += (dz / distance) * moveSpeed;
                } else {
                    robotState.x = targetX;
                    robotState.z = targetZ;
                    isMoving = false;
                }
            }

            robotGroup.position.x = robotState.x;
            robotGroup.position.z = robotState.z;
            robotGroup.rotation.y = THREE.MathUtils.degToRad(robotState.yaw);

            controls.target.set(robotState.x, 0, robotState.z);

            controls.update();
            renderer.render(scene, camera);
        }
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // Start WebSocket connections
        connectWebSocket();
        connectMockWebSocket();

        // Start animation
        animate();
    </script>
</body>
</html>
        """
        return HTMLResponse(content=html_content)

    async def broadcast_state(self):
        """Broadcast current robot state to all connected clients."""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(self.robot_state)
            except Exception as e:
                logging.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        for connection in disconnected:
            self.active_connections.discard(connection)

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-180, 180] range."""
        while angle < -180:
            angle += 360.0
        while angle > 180:
            angle -= 360.0
        return angle

    def tick(self) -> None:
        """Update simulator state."""
        if self._initialized:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            try:
                loop.run_until_complete(self.broadcast_state())
            except Exception as e:
                logging.error(f"Error in tick: {e}")

    def sim(self, actions: List[Action]) -> None:
        """Handle simulation updates from commands."""
        if not self._initialized:
            logging.warning("ThreeJSSim not initialized, skipping sim update")
            return

        try:
            with self._lock:
                for action in actions:
                    if action.type == "move":
                        self.robot_state["current_action"] = action.value
                        logging.info(f"ThreeJSSim received move action: {action.value}")
                    elif action.type == "speak":
                        logging.info(
                            f"ThreeJSSim received speak action: {action.value}"
                        )
                    elif action.type == "emotion":
                        logging.info(
                            f"ThreeJSSim received emotion action: {action.value}"
                        )

                # Broadcast updated state
                asyncio.create_task(self.broadcast_state())

        except Exception as e:
            logging.error(f"Error in sim update: {e}")

    async def cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up ThreeJSSim...")
        self._initialized = False

        for connection in list(self.active_connections):
            try:
                await connection.close()
            except Exception as e:
                logging.error(f"Error closing connection: {e}")
        self.active_connections.clear()
