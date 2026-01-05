import json
import logging
import os
from typing import Any, Dict
from uuid import uuid4

import dotenv
import json5
import zenoh
from jsonschema import ValidationError, validate

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

    Security Features:
    - Authentication via API key validation
    - Schema validation for all config updates
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
        self._authorized_api_key = self._load_authorized_api_key()

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

    def _load_authorized_api_key(self) -> str | None:
        dotenv.load_dotenv()
        api_key = os.getenv("OM_API_KEY")

        if not api_key and os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = json5.load(f)
                    api_key = config.get("api_key")
            except Exception as e:
                logging.warning(f"Could not load API key from config: {e}")

        if api_key:
            logging.info("ConfigProvider: API key authentication enabled")
        else:
            logging.warning(
                "ConfigProvider: No API key configured. Config updates will be rejected for security."
            )

        return api_key

    def _verify_api_key(self, provided_key: str | None) -> bool:
        if not self._authorized_api_key:
            logging.warning(
                "ConfigProvider: Rejecting config update - no authorized API key configured"
            )
            return False

        if not provided_key:
            logging.warning(
                "ConfigProvider: Rejecting config update - no API key provided"
            )
            return False

        return self._constant_time_compare(
            provided_key.encode("utf-8"), self._authorized_api_key.encode("utf-8")
        )

    def _constant_time_compare(self, a: bytes, b: bytes) -> bool:
        if len(a) != len(b):
            return False
        result = 0
        for x, y in zip(a, b):
            result |= x ^ y
        return result == 0

    def _detect_config_type(self, config: dict) -> str:
        if "modes" in config and "default_mode" in config:
            return "multi_mode"
        return "single_mode"

    def _validate_config_schema(self, config: dict) -> tuple[bool, str]:
        try:
            config_type = self._detect_config_type(config)
            schema_file = (
                "multi_mode_schema.json"
                if config_type == "multi_mode"
                else "single_mode_schema.json"
            )

            schema_path = os.path.join(
                os.path.dirname(__file__), "../../config/schema", schema_file
            )

            if not os.path.exists(schema_path):
                return False, f"Schema file not found: {schema_file}"

            with open(schema_path, "r") as f:
                schema = json.load(f)

            validate(instance=config, schema=schema)
            return True, ""

        except ValidationError as e:
            error_path = ".".join(str(p) for p in e.path) if e.path else "root"
            error_msg = f"Schema validation failed at '{error_path}': {e.message}"
            logging.error(f"ConfigProvider: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Schema validation error: {str(e)}"
            logging.error(f"ConfigProvider: {error_msg}")
            return False, error_msg

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
                # This is a set_config request - requires authentication
                # Extract API key from request if available
                # Note: ConfigRequest may need to be extended to include auth
                # For now, we'll check if API key is in the config itself
                self._handle_set_config(request.request_id, request.config.data)
            else:
                # This is a get_config request - no auth required for read
                self._send_config_response(request.request_id)

        except Exception as e:
            logging.error(f"Error handling config request: {e}")

    def _handle_set_config(self, request_id: String, config_str: str):
        """
        Handle request to update runtime configuration.

        Security checks:
        1. Parse and validate JSON5 syntax
        2. Verify API key authentication
        3. Validate against schema
        4. Write new config atomically
        """
        try:
            # Step 1: Parse JSON5
            new_config: Dict[str, Any]
            try:
                parsed = json5.loads(config_str)
                if not isinstance(parsed, dict):
                    error_msg = "Config must be a JSON object"
                    logging.error(f"ConfigProvider: {error_msg}")
                    self._send_error_response(request_id, error_msg)
                    return
                new_config = parsed
            except (ValueError, TypeError) as e:
                error_msg = f"Invalid JSON5 syntax: {str(e)}"
                logging.error(f"ConfigProvider: {error_msg}")
                self._send_error_response(request_id, error_msg)
                return
            except Exception as e:
                error_msg = f"Failed to parse config: {str(e)}"
                logging.error(f"ConfigProvider: {error_msg}")
                self._send_error_response(request_id, error_msg)
                return

            # Step 2: Verify API key authentication
            provided_api_key = new_config.get("api_key")
            if not self._verify_api_key(provided_api_key):
                error_msg = "Authentication failed: Invalid or missing API key"
                logging.warning(
                    f"ConfigProvider: {error_msg} (request_id: {request_id})"
                )
                self._send_error_response(request_id, error_msg)
                return

            # Step 3: Validate against schema
            is_valid, validation_error = self._validate_config_schema(new_config)
            if not is_valid:
                logging.warning(
                    f"ConfigProvider: Schema validation failed: {validation_error}"
                )
                self._send_error_response(
                    request_id, f"Schema validation failed: {validation_error}"
                )
                return

            # Step 4: Write new config atomically
            temp_path = self.config_path + ".tmp"

            try:
                with open(temp_path, "w") as f:
                    json.dump(new_config, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                os.rename(temp_path, self.config_path)

                logging.info(
                    f"ConfigProvider: Successfully updated config (request_id: {request_id})"
                )

                self._send_config_response(request_id)

            except Exception as e:
                error_msg = f"Failed to write config file: {str(e)}"
                logging.error(f"ConfigProvider: {error_msg}")
                self._send_error_response(request_id, error_msg)
                return

        except Exception as e:
            error_msg = f"Unexpected error updating config: {str(e)}"
            logging.error(f"ConfigProvider: {error_msg}")
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
