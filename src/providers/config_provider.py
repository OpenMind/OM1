import json
import logging
import os
import shutil
from uuid import uuid4

import json5
import zenoh
from jsonschema import ValidationError, validate

from runtime.version import verify_runtime_version
from zenoh_msgs import (
    ConfigRequest,
    ConfigResponse,
    String,
    open_zenoh_session,
    prepare_header,
)

from .singleton import singleton


@singleton
class ConfigProvider:
    """
    Singleton provider for runtime configuration broadcasting via Zenoh.
    """

    def __init__(self):
        """
        Initialize the ConfigProvider.
        """
        self.session = None
        self.config_response_publisher = None
        self.config_request_subscriber = None
        self.running = False

        self.config_path = self._get_runtime_config_path()

        self._initialize_zenoh()

    def _initialize_zenoh(self):
        """
        Initialize Zenoh session, publishers, and subscriber.
        """
        try:
            self.session = open_zenoh_session()

            # Publisher for config responses
            self.config_response_publisher = self.session.declare_publisher(
                "om/config/response"
            )

            # Subscriber for config requests
            self.config_request_subscriber = self.session.declare_subscriber(
                "om/config/request", self._handle_config_request
            )

            self.running = True
            logging.info("ConfigProvider initialized with Zenoh")
        except Exception as e:
            logging.error(f"Failed to initialize ConfigProvider Zenoh session: {e}")

    def _get_runtime_config_path(self) -> str:
        """
        Get the path to the runtime config file in memory folder.

        Returns
        -------
        str
            Path to config/memory/.runtime.json5
        """
        memory_folder_path = os.path.join(
            os.path.dirname(__file__), "../../config", "memory"
        )
        return os.path.abspath(os.path.join(memory_folder_path, ".runtime.json5"))

    def _handle_config_request(self, sample: zenoh.Sample):
        """
        Handle incoming config requests from Zenoh subscriber.

        Responds with current runtime configuration.

        Parameters
        ----------
        sample : zenoh.Sample
            The Zenoh sample containing the serialized ConfigRequest message.
        """
        try:
            request = ConfigRequest.deserialize(sample.payload.to_bytes())
            logging.debug(f"Received config request: {request.request_id}")

            if request.config and request.config.data:
                # This is a set_config request
                self._handle_set_config(request.request_id, request.config.data)
            else:
                # This is a get_config request
                self._send_config_response(request.request_id)

        except Exception as e:
            logging.error(f"Error handling config request: {e}")

    def _handle_set_config(self, request_id: String, config_str: str):
        """
        Handle request to update runtime configuration.

        """
        try:
            try:
                new_config = json5.loads(config_str)
            except (ValueError, json5.JSON5DecodeError) as e:
                error_msg = f"Invalid JSON5 syntax: {e}"
                logging.error(error_msg)
                self._send_error_response(request_id, error_msg)
                return
            except Exception as e:
                error_msg = f"Failed to parse configuration: {e}"
                logging.error(error_msg)
                self._send_error_response(request_id, error_msg)
                return

            is_multi_mode = "modes" in new_config and "default_mode" in new_config

            try:
                schema_file = (
                    "multi_mode_schema.json"
                    if is_multi_mode
                    else "single_mode_schema.json"
                )
                schema_path = os.path.join(
                    os.path.dirname(__file__),
                    "../../config/schema",
                    schema_file,
                )

                if not os.path.exists(schema_path):
                    error_msg = (
                        f"Schema file not found: {schema_path}. "
                        "Cannot validate configuration."
                    )
                    logging.error(error_msg)
                    self._send_error_response(request_id, error_msg)
                    return

                with open(schema_path, "r") as f:
                    schema = json.load(f)

                validate(instance=new_config, schema=schema)
                logging.debug(
                    f"Schema validation passed for {'multi-mode' if is_multi_mode else 'single-mode'} configuration"
                )

            except ValidationError as e:
                field_path = ".".join(str(p) for p in e.path) if e.path else "root"
                error_msg = (
                    f"Schema validation failed at field '{field_path}': {e.message}"
                )
                logging.error(error_msg)
                self._send_error_response(request_id, error_msg)
                return
            except Exception as e:
                error_msg = f"Schema validation error: {e}"
                logging.error(error_msg)
                self._send_error_response(request_id, error_msg)
                return

            try:
                config_version = new_config.get("version")
                verify_runtime_version(
                    config_version, config_name="runtime configuration update"
                )
                logging.debug(f"Version compatibility verified: {config_version}")
            except ValueError as e:
                error_msg = f"Version compatibility check failed: {e}"
                logging.error(error_msg)
                self._send_error_response(request_id, error_msg)
                return
            except Exception as e:
                error_msg = f"Version verification error: {e}"
                logging.error(error_msg)
                self._send_error_response(request_id, error_msg)
                return

            try:
                backup_path = None
                if os.path.exists(self.config_path):
                    backup_path = self.config_path + ".backup"
                    try:
                        shutil.copy2(self.config_path, backup_path)
                        logging.debug(f"Created backup: {backup_path}")
                    except Exception as backup_error:
                        logging.warning(
                            f"Failed to create backup: {backup_error}. "
                            "Continuing with update..."
                        )

                temp_path = self.config_path + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(new_config, f, indent=2)

                os.rename(temp_path, self.config_path)

                if backup_path and os.path.exists(backup_path):
                    try:
                        os.remove(backup_path)
                        logging.debug(f"Removed backup: {backup_path}")
                    except Exception:
                        pass

                logging.info(
                    f"Successfully updated runtime config file: {self.config_path}"
                )
                self._send_config_response(request_id)

            except OSError as e:
                error_msg = f"File system error while writing config: {e}"
                logging.error(error_msg)
                if backup_path and os.path.exists(backup_path):
                    try:
                        shutil.copy2(backup_path, self.config_path)
                        logging.info("Restored config from backup after write failure")
                    except Exception:
                        pass
                self._send_error_response(request_id, error_msg)
                return
            except Exception as e:
                error_msg = f"Unexpected error writing config file: {e}"
                logging.error(error_msg)
                self._send_error_response(request_id, error_msg)
                return

        except Exception as e:
            error_msg = f"Unexpected error in config update: {e}"
            logging.error(error_msg, exc_info=True)
            self._send_error_response(request_id, error_msg)

    def _send_config_response(self, request_id: String):
        """
        Send current runtime configuration as response.
        """
        try:
            # Get current config
            config_snapshot = self._get_config_snapshot()
            config_json_str = json.dumps(config_snapshot, indent=2)

            response = ConfigResponse(
                header=prepare_header(str(uuid4())),
                request_id=request_id,
                config=String(config_json_str),
                message=String("Configuration retrieved successfully"),
            )

            if self.config_response_publisher:
                self.config_response_publisher.put(response.serialize())
                logging.info("ConfigProvider sent config response")

        except Exception as e:
            logging.error(f"Failed to send config response: {e}")
            self._send_error_response(request_id, str(e))

    def _send_error_response(self, request_id: String, error_message: str):
        """
        Send error response.
        """
        try:
            response = ConfigResponse(
                header=prepare_header(str(uuid4())),
                request_id=request_id,
                config=String(""),
                message=String(error_message),
            )

            if self.config_response_publisher:
                self.config_response_publisher.put(response.serialize())
                logging.warning(f"ConfigProvider sent error response: {error_message}")

        except Exception as e:
            logging.error(f"Failed to send error response: {e}")

    def _get_config_snapshot(self) -> dict:
        """
        Get a snapshot of the current runtime configuration.
        """
        try:
            if not os.path.exists(self.config_path):
                logging.warning(
                    f"ConfigProvider: Config file not found: {self.config_path}"
                )
                return {}

            with open(self.config_path, "r") as f:
                return json5.load(f)

        except Exception as e:
            logging.error(f"Failed to read config file {self.config_path}: {e}")
            return {}

    def stop(self):
        """
        Stop the ConfigProvider and cleanup Zenoh session.
        """
        if not self.running:
            logging.info("ConfigProvider is not running")
            return

        self.running = False

        if self.session:
            self.session.close()

        logging.info("ConfigProvider stopped and Zenoh session closed")
