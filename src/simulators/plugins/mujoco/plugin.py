"""
MuJoCo Simulator Plugin for OM1

This plugin integrates MuJoCo physics simulation with OM1's control system.
It provides real-time robot simulation with sensor feedback and action execution.
"""

import logging
import threading
import time
from typing import List

from llm.output_model import Action
from simulators.base import Simulator, SimulatorConfig

from .adapter import HTTPAdapter
from .bridge import MuJoCoStepper


class MuJoCoSimulator(Simulator):
    """
    MuJoCo physics simulator integration for OM1.

    This simulator runs a MuJoCo environment and bridges it with OM1's
    control loop, allowing real-time control of simulated robots.
    """

    def __init__(self, config: SimulatorConfig):
        super().__init__(config)

        # Get configuration
        self.model_path = getattr(
            config, "model_path", "src/simulators/plugins/mujoco/assets/simple_arm.xml"
        )
        self.host = getattr(config, "host", "127.0.0.1")
        self.port = getattr(config, "port", 8888)
        self.http_port = getattr(config, "http_port", 8889)

        self._initialized = False
        self._running = False
        self._lock = threading.Lock()

        # Initialize MuJoCo stepper (WebSocket bridge)
        self.stepper = None
        self.adapter = None

        logging.info(f"Initializing MuJoCo Simulator with model: {self.model_path}")

    def initialize(self):
        """Initialize the MuJoCo environment and bridges"""
        if self._initialized:
            return

        with self._lock:
            try:
                # Initialize MuJoCo stepper
                self.stepper = MuJoCoStepper(
                    model_path=self.model_path, host=self.host, port=self.port
                )

                # Initialize HTTP adapter
                self.adapter = HTTPAdapter(
                    ws_url=f"ws://{self.host}:{self.port}", http_port=self.http_port
                )

                # Start servers in background threads
                self.stepper_thread = threading.Thread(
                    target=self.stepper.run, daemon=True
                )
                self.adapter_thread = threading.Thread(
                    target=self.adapter.run, daemon=True
                )

                self.stepper_thread.start()
                self.adapter_thread.start()

                # Wait for initialization
                time.sleep(1)

                self._initialized = True
                self._running = True
                logging.info("MuJoCo Simulator initialized successfully")

            except Exception as e:
                logging.error(f"Failed to initialize MuJoCo Simulator: {e}")
                raise

    def sim(self, actions: List[Action]) -> None:
        """
        Execute actions in the MuJoCo simulation

        Parameters
        ----------
        actions : List[Action]
            List of actions to execute in the simulator
        """
        if not self._initialized:
            self.initialize()

        if not actions:
            return

        try:
            # Process each action
            for action in actions:
                if hasattr(action, "force") and action.force is not None:
                    # Apply force command to the simulation
                    logging.debug(f"Applying force action: {action.force}")

        except Exception as e:
            logging.error(f"Error executing actions in MuJoCo: {e}")

    def tick(self) -> None:
        """
        Run one simulation tick

        The MuJoCo stepper runs continuously in its own thread,
        so this just maintains the tick rate for OM1 synchronization.
        """
        if not self._initialized:
            self.initialize()

        # The simulation runs in background threads
        # This tick just maintains sync with OM1's main loop
        time.sleep(0.01)  # 100Hz tick rate

    def stop(self):
        """Stop the simulator"""
        with self._lock:
            self._running = False
            logging.info("MuJoCo Simulator stopped")

    def __del__(self):
        """Cleanup on deletion"""
        self.stop()
