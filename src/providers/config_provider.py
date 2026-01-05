import json
import logging
import os
import shutil
from typing import Any, Dict
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
        self.session = None
        self.config_response_publisher = None
        self.config_request_subscriber = None
        self.running = False

        self.config_path = self._get_runtime_config_path()
        self._initialize_zenoh()

    def _initialize_zenoh(self):
        """Initialize the Zenoh session and publishers/subscribers."""
        try:
            self.session = open_zenoh_session()

            self.config_response_publisher = self.session.declare_publisher(
                "om/config/response"
            )

            self.config_request_subscriber = self.session.declare_subscriber(
                "om/config/request", self._handle_config_request
            )

            self.running = True
            logging.info("ConfigProvider initialized with Zenoh")
        except Exception as e:
            logging.error(f"Failed to initialize ConfigProvider Zenoh session: {e}")

    def _get_runtime_config_path(self) -> str:
        """Return the absolute path of the runtime configuration file."""
        memory_folder_path = os.path.join(
            os.path.dirname(__file__), "../../config", "memory"
        )
        return os.path.abspath(os.path.join(memory_folder_path, ".runtime.json5"))

    def _handle_config_request(self, sample: zenoh.Sample):
        """Handle incoming configuration requests."""
        try:
            request = ConfigRequest.deserialize(sample.payload.to_bytes())
            logging.debug(f"Received config request: {request.request_id}")

            if request.config and request.config.data:
                self._handle_set_config(request.request_id, request.config.data)
            else:
                self._send_config_response(request.request_id)

        except Exception as e:
            logging.error(f"Error handling config request: {e}")

    def _handle_set_config(self, request_id: String, config_str: str):
        """Handle requests to update the runtime configuration."""
        try:
            try:
                new_config: Any = json5.loads(config_str)
                if not isinstance(new_config, dict):
                    raise ValueError("Configuration root must be a JSON object")

            except ValueError as e:
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
                    error_msg = f"Schema file not found: {schema_path}"
                    logging.error(error_msg)
                    self._send_error_response(request_id, error_msg)
                    return

                with open(schema_path, "r") as f:
                    schema = json.load(f)

                validate(instance=new_config, schema=schema)

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
            except Exception as e:
                error_msg = f"Version verification error: {e}"
                logging.error(error_msg)
                self._send_error_response(request_id, error_msg)
                return

            backup_path: str | None = None

            try:
                if os.path.exists(self.config_path):
                    backup_path = self.config_path + ".backup"
                    shutil.copy2(self.config_path, backup_path)

                temp_path = self.config_path + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(new_config, f, indent=2)

                os.replace(temp_path, self.config_path)

                if backup_path and os.path.exists(backup_path):
                    os.remove(backup_path)

                self._send_config_response(request_id)

            except OSError as e:
                error_msg = f"File system error while writing config: {e}"
                logging.error(error_msg)

                if backup_path and os.path.exists(backup_path):
                    shutil.copy2(backup_path, self.config_path)

                self._send_error_response(request_id, error_msg)

        except Exception as e:
            error_msg = f"Unexpected error in config update: {e}"
            logging.error(error_msg, exc_info=True)
            self._send_error_response(request_id, error_msg)

    def _send_config_response(self, request_id: String):
        """Send the current configuration snapshot as a response."""
        try:
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

        except Exception as e:
            logging.error(f"Failed to send config response: {e}")
            self._send_error_response(request_id, str(e))

    def _send_error_response(self, request_id: String, error_message: str):
        """Send an error response for a configuration request."""
        try:
            response = ConfigResponse(
                header=prepare_header(str(uuid4())),
                request_id=request_id,
                config=String(""),
                message=String(error_message),
            )

            if self.config_response_publisher:
                self.config_response_publisher.put(response.serialize())

        except Exception as e:
            logging.error(f"Failed to send error response: {e}")

    def _get_config_snapshot(self) -> Dict[str, Any]:
        """Return the current runtime configuration snapshot."""
        try:
            if not os.path.exists(self.config_path):
                return {}

            with open(self.config_path, "r") as f:
                data = json5.load(f)

            return data if isinstance(data, dict) else {}

        except Exception as e:
            logging.error(f"Failed to read config file {self.config_path}: {e}")
            return {}

    def stop(self):
        """Stop the ConfigProvider and close the Zenoh session."""
        if not self.running:
            return

        self.running = False

        if self.session:
            self.session.close()
