import asyncio
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from llm.output_model import Action
from providers.io_provider import Input, IOProvider
from simulators.base import Simulator, SimulatorConfig


@dataclass
class SimulatorState:
    """Dataclass representing the state of the simulator."""

    inputs: dict
    current_action: str = "idle"
    last_speech: str = ""
    current_emotion: str = ""
    system_latency: Optional[dict] = None

    def to_dict(self):
        """
        Convert the SimulatorState to a dictionary.
        """
        return asdict(self)


class WebSim(Simulator):
    """
    WebSim simulator class for visualizing simulation data in a web interface.
    """

    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        self.messages: list[str] = []
        self.io_provider = IOProvider()

        self._initialized = False
        self._lock = threading.Lock()
        self._last_tick = time.time()
        self._tick_interval = 0.1  # 100ms tick rate

        self.state_dict = {}
        # Initialize state
        self.state = SimulatorState(
            inputs={},
            current_action="idle",
            last_speech="",
            current_emotion="",
            system_latency={
                "fuse_time": 0,
                "llm_start": 0,
                "processing": 0,
                "complete": 0,
            },
        )

        logging.info("Initializing WebSim...")

        # Initialize FastAPI app
        self.app = FastAPI()

        # Mount assets directory
        assets_path = os.path.join(os.path.dirname(__file__), "assets")
        if not os.path.exists(assets_path):
            os.makedirs(assets_path)
        self.app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

        # Ensure the logo exists in assets directory
        logo_path = os.path.join(assets_path, "OM_Logo_b_transparent.png")
        if not os.path.exists(logo_path):
            logging.warning(f"Logo not found at {logo_path}")

        self.active_connections: List[WebSocket] = []

        # Setup routes
        @self.app.get("/")
        async def get_index():
            return HTMLResponse(
                """
            <!DOCTYPE html>
            <html>
                <head>
                    <title>OpenMind Simulator</title>
                    <script src="https://unpkg.com/react@17/umd/react.development.js"></script>
                    <script src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
                    <script src="https://unpkg.com/babel-standalone@6/babel.min.js"></script>
                    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
                    <style>
                        .message-header {
                            cursor: pointer;
                            padding: 8px;
                            border-radius: 4px;
                            background-color: #f3f4f6;
                            transition: background-color 0.2s;
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                        }
                        .message-header:hover {
                            background-color: #e5e7eb;
                        }
                        .message-preview {
                            overflow: hidden;
                            text-overflow: ellipsis;
                            white-space: nowrap;
                            color: #6B7280;
                            font-size: 0.875rem;
                            flex: 1;
                            margin: 0 8px;
                            position: relative;
                            padding-right: 24px;
                        }
                        .message-arrow {
                            color: #6B7280;
                            font-size: 0.875rem;
                            min-width: 20px;
                            text-align: center;
                            transform: rotate(0deg);
                            transition: transform 0.2s;
                        }
                        .message-arrow.expanded {
                            transform: rotate(90deg);
                        }
                        .resize-handle {
                            color: #9CA3AF;
                            font-size: 1rem;
                            cursor: col-resize;
                            padding: 0 4px;
                            position: absolute;
                            right: 0;
                            top: 50%;
                            transform: translateY(-50%);
                        }
                        .message-timestamp {
                            color: #9CA3AF;
                            font-size: 0.75rem;
                            min-width: 60px;
                        }
                        .footer {
                            position: fixed;
                            bottom: 0;
                            left: 0;
                            right: 0;
                            padding: 1rem;
                            background-color: white;
                            border-top: 1px solid #e5e7eb;
                            display: flex;
                            align-items: center;
                            justify-content: space-between;
                        }
                        .footer-logo {
                            height: 60px;
                            width: auto;
                            margin-left: 1rem;
                        }
                        .footer-links {
                            display: flex;
                            gap: 2rem;
                            margin-right: 2rem;
                        }
                        .footer-link {
                            color: #3B82F6;
                            text-decoration: none;
                            font-size: 1rem;
                            font-weight: 500;
                            transition: color 0.2s;
                            display: flex;
                            align-items: center;
                            gap: 0.5rem;
                        }
                        .footer-link:hover {
                            color: #2563EB;
                        }
                        .github-icon {
                            width: 20px;
                            height: 20px;
                        }
                        .message-content {
                            position: relative;
                            margin-top: 0.5rem;
                            background-color: #f9fafb;
                            border-radius: 0.375rem;
                            overflow: hidden;
                        }
                        .message-text {
                            white-space: pre-wrap;
                            word-break: break-word;
                            font-size: 0.875rem;
                            color: #374151;
                            padding: 0.5rem 1rem;
                            overflow-y: auto;
                        }
                        .content-resize-handle {
                            position: absolute;
                            right: 4px;
                            bottom: 4px;
                            width: 12px;
                            height: 12px;
                            cursor: nw-resize;
                            opacity: 0.6;
                            background-image: radial-gradient(circle, #9CA3AF 1.5px, transparent 1.5px);
                            background-size: 4px 4px;
                            transition: opacity 0.2s;
                        }
                        .content-resize-handle:hover {
                            opacity: 1;
                        }
                    </style>
                </head>
                <body class="bg-gray-50">
                    <div id="root"></div>
                    <script type="text/babel">
                        function App() {
                            const [state, setState] = React.useState({
                                inputs: {},
                                current_action: "idle",
                                last_speech: "",
                                current_emotion: "",
                                system_latency: {
                                    fuse_time: 0,
                                    llm_start: 0,
                                    processing: 0,
                                    complete: 0
                                }
                            });
                            const [error, setError] = React.useState(null);
                            const [connected, setConnected] = React.useState(false);
                            const [expandedMessages, setExpandedMessages] = React.useState({});
                            const [messageHeights, setMessageHeights] = React.useState({});
                            const [isResizing, setIsResizing] = React.useState(null);
                            const resizeRef = React.useRef(null);

                            const startResizing = React.useCallback((messageId, e) => {
                                e.stopPropagation();
                                setIsResizing(messageId);
                                document.body.classList.add('no-select');
                            }, []);

                            const stopResizing = React.useCallback(() => {
                                setIsResizing(null);
                                document.body.classList.remove('no-select');
                            }, []);

                            const handleResize = React.useCallback((e) => {
                                if (isResizing && resizeRef.current) {
                                    const container = resizeRef.current;
                                    const containerRect = container.getBoundingClientRect();
                                    const newHeight = Math.max(100, e.clientY - containerRect.top);
                                    setMessageHeights(prev => ({
                                        ...prev,
                                        [isResizing]: newHeight
                                    }));
                                }
                            }, [isResizing]);

                            React.useEffect(() => {
                                const ws = new WebSocket(`ws://${window.location.host}/ws`);

                                ws.onopen = () => {
                                    console.log('Connected to WebSocket');
                                    setConnected(true);
                                };

                                ws.onmessage = (event) => {
                                    const data = JSON.parse(event.data);
                                    setState(data);
                                };

                                ws.onerror = (error) => {
                                    console.error('WebSocket error:', error);
                                    setError('Failed to connect to server');
                                };

                                ws.onclose = () => {
                                    setConnected(false);
                                    setError('Connection lost. Reconnecting...');
                                    setTimeout(() => window.location.reload(), 2000);
                                };

                                return () => ws.close();
                            }, []);

                            React.useEffect(() => {
                                if (isResizing) {
                                    window.addEventListener('mousemove', handleResize);
                                    window.addEventListener('mouseup', stopResizing);
                                    return () => {
                                        window.removeEventListener('mousemove', handleResize);
                                        window.removeEventListener('mouseup', stopResizing);
                                    };
                                }
                            }, [isResizing, handleResize, stopResizing]);

                            const groupedMessages = React.useMemo(() => {
                                const groups = {};
                                Object.entries(state.inputs || {}).forEach(([key, value]) => {
                                    const inputType = value.input_type || 'Unknown';
                                    if (!groups[inputType]) {
                                        groups[inputType] = [];
                                    }
                                    groups[inputType].push({ id: key, ...value });
                                });
                                return groups;
                            }, [state.inputs]);

                            if (error) {
                                return (
                                    <div className="min-h-screen flex items-center justify-center">
                                        <div className="text-red-600">{error}</div>
                                    </div>
                                );
                            }

                            if (!connected) {
                                return (
                                    <div className="min-h-screen flex items-center justify-center">
                                        <div>Connecting...</div>
                                    </div>
                                );
                            }

                            return (
                                <div className="min-h-screen p-4 pb-16">
                                    <div className="container mx-auto">
                                        <div className="flex">
                                            {/* Input History */}
                                            <div className="bg-white rounded-lg shadow p-4" style={{ width: '33%' }}>
                                                <div className="flex flex-col">
                                                    <h2 className="text-xl font-bold mb-4">Input History</h2>
                                                    <div className="space-y-2">
                                                        {Object.entries(groupedMessages)
                                                            .sort(([a], [b]) => a.localeCompare(b))
                                                            .map(([inputType, messages]) => (
                                                                <div key={inputType}>
                                                                    <h3 className="text-sm font-semibold text-gray-700 mb-2">
                                                                        {inputType}
                                                                    </h3>
                                                                    {messages
                                                                        .sort((a, b) => b.timestamp - a.timestamp)
                                                                        .map((message) => (
                                                                            <div key={message.id} className="mb-2">
                                                                                <div
                                                                                    className="message-header"
                                                                                    onClick={() => setExpandedMessages(prev => ({
                                                                                        ...prev,
                                                                                        [message.id]: !prev[message.id]
                                                                                    }))}
                                                                                >
                                                                                    <span className={`message-arrow ${expandedMessages[message.id] ? 'expanded' : ''}`}>
                                                                                        ▶
                                                                                    </span>
                                                                                    <span className="message-timestamp">
                                                                                        {message.timestamp.toFixed(3)}s
                                                                                    </span>
                                                                                    <span className="message-preview">
                                                                                        {message.input.substring(0, 50)}
                                                                                        {message.input.length > 50 ? '...' : ''}
                                                                                    </span>
                                                                                </div>
                                                                                {expandedMessages[message.id] && (
                                                                                    <div
                                                                                        className="message-content"
                                                                                        ref={isResizing === message.id ? resizeRef : null}
                                                                                        style={{ height: messageHeights[message.id] || 'auto', minHeight: '100px' }}
                                                                                    >
                                                                                        <div className="message-text">
                                                                                            {message.input}
                                                                                        </div>
                                                                                        <div
                                                                                            className="content-resize-handle"
                                                                                            onMouseDown={(e) => startResizing(message.id, e)}
                                                                                        />
                                                                                    </div>
                                                                                )}
                                                                            </div>
                                                                        ))}
                                                                </div>
                                                            ))}
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Main Display */}
                                            <div className="flex-1 ml-4">
                                                <div className="bg-white rounded-lg shadow p-4 mb-4">
                                                    <h2 className="text-xl font-bold mb-4">Current State</h2>
                                                    <div className="space-y-4">
                                                        <div>
                                                            <span className="font-semibold">Action:</span>
                                                            <span className="ml-2 text-blue-600">{state.current_action}</span>
                                                        </div>
                                                        <div>
                                                            <span className="font-semibold">Last Speech:</span>
                                                            <div className="mt-1 p-2 bg-gray-50 rounded">
                                                                {state.last_speech || "No speech yet"}
                                                            </div>
                                                        </div>
                                                        <div>
                                                            <span className="font-semibold">Emotion:</span>
                                                            <span className="ml-2 text-purple-600">{state.current_emotion}</span>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="bg-white rounded-lg shadow p-4">
                                                    <h2 className="text-xl font-bold mb-4">System Latency</h2>
                                                    <div className="space-y-2">
                                                        {Object.entries(state.system_latency || {}).map(([key, value]) => (
                                                            <div key={key} className="flex justify-between items-center">
                                                                <span className="font-semibold">{key}:</span>
                                                                <span className="text-gray-600">
                                                                    {value ? value.toFixed(3) : '0.000'}s
                                                                </span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="footer">
                                        <img
                                            src="/assets/OM_Logo_b_transparent.png"
                                            alt="OpenMind Logo"
                                            className="footer-logo"
                                        />
                                        <div className="footer-links">
                                            <a
                                                href="https://github.com/OpenmindAGI/OM1"
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="footer-link"
                                            >
                                                <svg className="github-icon" viewBox="0 0 24 24" fill="currentColor">
                                                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                                                </svg>
                                                GitHub
                                            </a>
                                            <a
                                                href="https://docs.openmind.org/introduction"
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="footer-link"
                                            >
                                                Documentation
                                            </a>
                                        </div>
                                    </div>
                                </div>
                            );
                        }

                        ReactDOM.render(<App />, document.getElementById('root'));
                    </script>
                </body>
            </html>
            """
            )

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                await websocket.send_json(self.state.to_dict())
                while True:
                    await websocket.receive_text()
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
            finally:
                self.active_connections.remove(websocket)

        # Start server thread
        try:
            logging.info("Starting WebSim server thread...")
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            time.sleep(1)

            if self.server_thread.is_alive():
                # Using ANSI color codes for cyan text and bold
                logging.info(
                    "\033[1;36mWebSim server started successfully - Open http://localhost:8000 in your browser\033[0m"
                )
                self._initialized = True
            else:
                logging.error("WebSim server failed to start")
        except Exception as e:
            logging.error(f"Error starting WebSim server thread: {e}")

    def _run_server(self):
        """Run the FastAPI server."""
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",  # Still bind to all interfaces
            port=8000,
            log_level="error",
            server_header=False,
            # Override the default startup message
            log_config={
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "default": {
                        "()": "uvicorn.logging.DefaultFormatter",
                        "fmt": "%(message)s",
                    },
                },
                "handlers": {
                    "default": {
                        "formatter": "default",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stderr",
                    },
                },
                "loggers": {
                    "uvicorn": {"handlers": ["default"], "level": "ERROR"},
                    "uvicorn.error": {"level": "ERROR"},
                },
            },
        )
        server = uvicorn.Server(config)
        server.run()

    async def broadcast_state(self):
        """Broadcast current state to all connected clients."""
        if not self.active_connections:
            return

        try:

            # Broadcast to all clients
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(self.state_dict)
                except Exception as e:
                    logging.error(f"Error broadcasting to client: {e}")
                    disconnected.append(connection)

            for connection in disconnected:
                try:
                    self.active_connections.remove(connection)
                except ValueError:
                    pass

        except Exception as e:
            logging.error(f"Error in broadcast_state: {e}")

    def get_earliest_time(self, inputs: Dict[str, Input]) -> float:
        """Get earliest timestamp from inputs."""
        earliest_time = float("inf")
        for input_type, input_info in inputs.items():
            logging.debug(f"GET {input_info}")
            if input_type == "GovernanceEthereum":
                continue
            if input_type == "Universal Laws":
                continue
            if input_info.timestamp is not None:
                if input_info.timestamp < earliest_time:
                    earliest_time = float(input_info.timestamp)
        return earliest_time if earliest_time != float("inf") else 0.0

    def tick(self) -> None:
        """Update simulator state."""
        if self._initialized:
            try:
                # Get or create event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                try:
                    loop.run_until_complete(self.broadcast_state())
                except Exception:
                    loop = asyncio.get_event_loop()
                    loop.create_task(self.broadcast_state())

            except Exception as e:
                logging.error(f"Error in tick: {e}")

            time.sleep(0.5)

    def sim(self, actions: List[Action]) -> None:
        """Handle simulation updates from commands."""
        if not self._initialized:
            logging.warning("WebSim not initialized, skipping sim update")
            return

        try:
            updated = False
            with self._lock:
                earliest_time = self.get_earliest_time(self.io_provider.inputs)
                logging.debug(f"earliest_time: {earliest_time}")

                input_rezeroed = []
                for input_type, input_info in self.io_provider.inputs.items():
                    timestamp = 0
                    if (
                        input_type != "GovernanceEthereum"
                        and input_info.timestamp is not None
                    ):
                        timestamp = input_info.timestamp - earliest_time
                    input_rezeroed.append(
                        {
                            "input_type": input_type,
                            "timestamp": timestamp,
                            "input": input_info.input,
                        }
                    )

                # Process system latency relative to earliest time
                fuser_end_time = self.io_provider.fuser_end_time or 0
                llm_start_time = self.io_provider.llm_start_time or 0
                llm_end_time = self.io_provider.llm_end_time or 0

                system_latency = {
                    "fuse_time": (
                        fuser_end_time - earliest_time if fuser_end_time else 0
                    ),
                    "llm_start": (
                        llm_start_time - earliest_time if llm_start_time else 0
                    ),
                    "processing": (
                        llm_end_time - llm_start_time
                        if (llm_end_time and llm_start_time)
                        else 0
                    ),
                    "complete": llm_end_time - earliest_time if llm_end_time else 0,
                }

                for action in actions:
                    if action.type == "move":
                        new_action = action.value
                        if new_action != self.state.current_action:
                            self.state.current_action = new_action
                            updated = True
                    elif action.type == "speak":
                        new_speech = action.value
                        if new_speech != self.state.last_speech:
                            self.state.last_speech = new_speech
                            updated = True
                    elif action.type == "emotion":
                        new_emotion = action.value
                        if new_emotion != self.state.current_emotion:
                            self.state.current_emotion = new_emotion
                            updated = True

                self.state_dict = {
                    "current_action": self.state.current_action,
                    "last_speech": self.state.last_speech,
                    "current_emotion": self.state.current_emotion,
                    "system_latency": system_latency,
                    "inputs": input_rezeroed,
                }

                logging.info(f"Inputs and LLM Outputs: {self.state_dict}")

            if updated:
                self._last_tick = 0
                self.tick()

        except Exception as e:
            logging.error(f"Error in sim update: {e}")

    async def cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up WebSim...")
        self._initialized = False

        for connection in self.active_connections[:]:
            try:
                await connection.close()
            except Exception as e:
                logging.error(f"Error closing connection: {e}")
        self.active_connections.clear()
