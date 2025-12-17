import asyncio
import logging
import time
from queue import Queue
from typing import List, Optional

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from inputs.plugins.rplidar import Message, RPLidar, RPLidarConfig
from providers.io_provider import IOProvider
from providers.rplidar_provider import RPLidarProvider
from tests.integration.mock_inputs.data_providers.mock_lidar_scan_provider import (
    get_lidar_provider,
    get_next_lidar_scan,
)


class MockRPLidar(RPLidar):
    """
    Mock implementation of RPLidar that uses mock lidar data.

    This class reuses the real RPLidarProvider and its path processing logic,
    but overrides the hardware interface to inject mock data instead of
    connecting to real hardware.
    """

    def __init__(self, config: RPLidarConfig = RPLidarConfig()):
        """
        Initialize with mock lidar provider that reuses real processing logic.

        Parameters
        ----------
        config : RPLidarConfig, optional
            Configuration for the sensor
        """
        # Initialize base FuserInput class, skipping RPLidar.__init__ to avoid hardware setup
        super(FuserInput, self).__init__(config)

        # Override the descriptor to indicate this is a mock
        self.descriptor_for_LLM = "MOCK RPLidar INPUT (Integration Test)"

        # Track IO
        self.io_provider = IOProvider()

        # Buffer for storing the final output
        self.messages: List[Message] = []

        # Buffer for storing messages
        self.message_buffer: Queue[str] = Queue()

        # Extract lidar configuration
        lidar_config = self._extract_lidar_config(config)

        # Create real RPLidarProvider but prevent hardware connections
        self.lidar: RPLidarProvider = self._create_mock_lidar_provider(**lidar_config)

        # Store the last processed time to rate-limit our mock data
        self.last_processed_time = 0

        # Track if we've processed all mock data
        self.mock_data_processed = False

        # Start the mock data processing
        self.running = True

        # Get reference to the mock lidar data provider
        self.lidar_provider = get_lidar_provider()

        # Store reference to cortex runtime for cleanup (will be set by test runner)
        self.cortex_runtime = None

        logging.info(
            f"MockRPLidar initialized with {self.lidar_provider.scan_count} scan data sets"
        )

    def set_cortex_runtime(self, cortex):
        """
        Set reference to cortex runtime for comprehensive cleanup.

        Parameters
        ----------
        cortex : CortexRuntime
            The cortex runtime instance
        """
        self.cortex_runtime = cortex

    def _create_mock_lidar_provider(self, **lidar_config) -> RPLidarProvider:
        """
        Create a RPLidarProvider instance but override its start method to prevent hardware connections.

        Returns
        -------
        RPLidarProvider
            A configured RPLidarProvider instance that won't try to connect to hardware
        """
        # Create the real provider with configuration
        provider = RPLidarProvider(**lidar_config)

        def mock_start():
            """Mock start method that doesn't try to connect to hardware."""
            provider.running = True
            logging.info(
                "MockRPLidar: RPLidarProvider start() called (hardware connections prevented)"
            )

        provider.start = mock_start

        # Override the stop method to be a no-op for mock
        def mock_stop():
            """Mock stop method for graceful cleanup."""
            provider.running = False
            provider.d435_provider.stop()
            logging.info("MockRPLidar: RPLidarProvider stop() called (no-op for mock)")

        provider.stop = mock_stop

        return provider

    def _extract_lidar_config(self, config: SensorConfig) -> dict:
        """Extract lidar configuration parameters from sensor config."""
        lidar_config = {
            "serial_port": getattr(config, "serial_port", None),
            "use_zenoh": getattr(config, "use_zenoh", False),
            "half_width_robot": getattr(config, "half_width_robot", 0.20),
            "angles_blanked": getattr(config, "angles_blanked", []),
            "relevant_distance_max": getattr(config, "relevant_distance_max", 1.1),
            "relevant_distance_min": getattr(config, "relevant_distance_min", 0.08),
            "sensor_mounting_angle": getattr(config, "sensor_mounting_angle", 180.0),
        }

        # Handle Zenoh-specific configuration
        if lidar_config["use_zenoh"]:
            lidar_config["URID"] = getattr(config, "URID", "default")
            logging.info(f"MockRPLidar using Zenoh with URID: {lidar_config['URID']}")

        return lidar_config

    async def _poll(self) -> Optional[str]:
        """
        Override the poll method to inject mock data into the real path processor.

        Returns
        -------
        Optional[str]
            Mock lidar data string if available, None otherwise
        """
        await asyncio.sleep(0.2)  # Maintain the same polling rate

        # Rate limit to avoid overwhelming the system
        current_time = time.time()
        if current_time - self.last_processed_time < 0.5:  # Half second between data
            return None

        self.last_processed_time = current_time

        # Get next scan from the mock provider
        scan_array = get_next_lidar_scan()
        if scan_array is not None:
            # Inject mock data into the real path processor
            # scan_array is already in the format (angle, distance) that _path_processor expects
            self.lidar._path_processor(scan_array)

            logging.info(
                f"MockRPLidar: Processed mock scan ({self.lidar_provider.remaining_scans} remaining)"
            )

            # Return the processed lidar string from the real provider
            return self.lidar.lidar_string
        else:
            if not self.mock_data_processed:
                logging.info("MockRPLidar: No more mock scan data to process")
                self.mock_data_processed = True
            return None

    # TODO (Kyle): Replace the odometer with a mock input to avoid clean up.
    async def cleanup_cortex_runtime(self):
        """
        Clean up CortexRuntime and its components to prevent hanging.
        This method handles the Zenoh session cleanup that's specific to RPLidar tests.
        """
        if not self.cortex_runtime:
            logging.warning("MockRPLidar: No cortex runtime reference for cleanup")
            return

        logging.info("MockRPLidar: Starting cortex cleanup")

        try:
            cortex = self.cortex_runtime

            # Clean up action orchestrator
            if hasattr(cortex, "action_orchestrator") and cortex.action_orchestrator:
                logging.info("MockRPLidar: Cleaning up action orchestrator")

                if (
                    hasattr(cortex.action_orchestrator, "_config")
                    and cortex.action_orchestrator._config
                ):
                    if hasattr(cortex.action_orchestrator._config, "agent_actions"):
                        agent_actions = cortex.action_orchestrator._config.agent_actions

                        for agent_action in agent_actions:
                            if (
                                hasattr(agent_action, "connector")
                                and agent_action.connector
                            ):
                                # Close Zenoh sessions in action connectors
                                if (
                                    hasattr(agent_action.connector, "session")
                                    and agent_action.connector.session
                                ):
                                    try:
                                        agent_action.connector.session.close()
                                    except Exception as e:
                                        logging.warning(
                                            f"MockRPLidar: Error closing Zenoh session: {e}"
                                        )

                                # Also close OdomProvider Zenoh session if it exists
                                if (
                                    hasattr(agent_action.connector, "odom")
                                    and agent_action.connector.odom
                                ):
                                    if (
                                        hasattr(agent_action.connector.odom, "session")
                                        and agent_action.connector.odom.session
                                    ):
                                        try:
                                            agent_action.connector.odom.session.close()
                                        except Exception as e:
                                            logging.warning(
                                                f"MockRPLidar: Error closing OdomProvider Zenoh session: {e}"
                                            )

            # Clean up any other Zenoh sessions in the cortex config
            if hasattr(cortex, "config") and cortex.config:
                # Check for any providers that might have Zenoh sessions
                if hasattr(cortex.config, "agent_inputs"):
                    for input_obj in cortex.config.agent_inputs:
                        if hasattr(input_obj, "lidar") and input_obj.lidar:
                            if hasattr(input_obj.lidar, "zen") and input_obj.lidar.zen:
                                try:
                                    input_obj.lidar.zen.close()
                                except Exception as e:
                                    logging.warning(
                                        f"MockRPLidar: Error closing lidar Zenoh session: {e}"
                                    )

            # Force cleanup of any remaining Zenoh sessions
            await self._force_cleanup_zenoh_sessions()

            logging.info("MockRPLidar: Cortex cleanup completed")

        except Exception as e:
            logging.error(f"MockRPLidar: Error during cortex cleanup: {e}")

    async def _force_cleanup_zenoh_sessions(self):
        """
        Force cleanup of any remaining Zenoh sessions by attempting to close them.
        """
        try:
            # Try to import zenoh and close any open sessions

            # Force garbage collection and wait for threads
            import gc

            gc.collect()

            # Wait a bit for any cleanup to complete
            await asyncio.sleep(0.5)

            # Check remaining threads
            import threading

            # Log remaining non-daemon threads only if there are any
            non_daemon_threads = []
            for thread in threading.enumerate():
                if thread != threading.current_thread() and not thread.daemon:
                    non_daemon_threads.append(thread)

            if non_daemon_threads:
                logging.warning(
                    f"MockRPLidar: {len(non_daemon_threads)} non-daemon threads still active"
                )

                # Try to force-kill the Zenoh threads by setting them as daemon
                for thread in non_daemon_threads:
                    if "pyo3-closure" in thread.name or "zenoh" in thread.name.lower():
                        try:
                            thread.daemon = True
                        except Exception:
                            pass  # Ignore errors when setting daemon

                # Give them a bit more time to finish
                await asyncio.sleep(1.0)

        except Exception as e:
            logging.warning(f"MockRPLidar: Error during Zenoh cleanup: {e}")

    def cleanup(self):
        """
        Synchronous cleanup method for proper resource cleanup.
        """
        logging.info("MockRPLidar.cleanup: Starting cleanup")

        try:
            self.running = False

            # Clean up the lidar provider
            if hasattr(self, "lidar") and self.lidar:
                self.lidar.stop()

            logging.info("MockRPLidar.cleanup: Cleanup completed successfully")
        except Exception as e:
            logging.error(f"MockRPLidar.cleanup: Error during cleanup: {e}")

    async def async_cleanup(self):
        """
        Asynchronous cleanup method that handles both basic cleanup and cortex runtime cleanup.
        """
        # First do the basic cleanup
        self.cleanup()

        # Then do the comprehensive cortex cleanup
        await self.cleanup_cortex_runtime()

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.cleanup()

    def reset_mock_data(self):
        """
        Reset the mock data to start from the beginning.
        Useful for repeated testing.
        """
        self.lidar_provider.reset()
        self.mock_data_processed = False
        logging.info("MockRPLidar: Mock data reset")
