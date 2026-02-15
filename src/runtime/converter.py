import logging
from typing import Dict


class ConfigConverter:
    """Convert single-mode configurations to multi-mode format."""

    @staticmethod
    def is_single_mode(raw_config: dict) -> bool:
        """Detect whether the configuration is in single-mode format.

        Parameters
        ----------
        raw_config : dict
            The raw configuration dictionary to check.

        Returns
        -------
        bool
            True if the config is single-mode (missing 'modes' or 'default_mode').
        """
        return "modes" not in raw_config or "default_mode" not in raw_config

    @staticmethod
    def convert_to_multi_mode(raw_config: dict) -> dict:
        """Convert a single-mode config to multi-mode format.

        If the config is already multi-mode, return it unchanged.

        Parameters
        ----------
        raw_config : dict
            The raw configuration dictionary to convert.

        Returns
        -------
        dict
            A multi-mode formatted configuration dictionary.
        """
        if not ConfigConverter.is_single_mode(raw_config):
            return raw_config

        mode_name = raw_config.get("name", "default")
        logging.info(f"Converting single-mode config '{mode_name}'")

        converted_config = ConfigConverter._build_global_section(raw_config, mode_name)
        converted_config["modes"] = {
            mode_name: ConfigConverter._build_mode_section(raw_config)
        }
        converted_config["transition_rules"] = []

        ConfigConverter._validate(converted_config, mode_name)

        return converted_config

    @staticmethod
    def _build_global_section(raw_config: dict, mode_name: str) -> Dict:
        """Build the global fields of a multi-mode config.

        Parameters
        ----------
        raw_config : dict
            The original single-mode configuration.
        mode_name : str
            Name of the mode to use as default.

        Returns
        -------
        dict
            Global-level fields for the multi-mode config.
        """
        return {
            "version": raw_config.get("version"),
            "name": mode_name,
            "default_mode": mode_name,
            "allow_manual_switching": False,
            "mode_memory_enabled": False,
            "api_key": raw_config.get("api_key", ""),
            "robot_ip": raw_config.get("robot_ip", ""),
            "URID": raw_config.get("URID", "default"),
            "unitree_ethernet": raw_config.get("unitree_ethernet", ""),
            "system_governance": raw_config.get("system_governance", ""),
            "system_prompt_examples": raw_config.get("system_prompt_examples", ""),
            "cortex_llm": raw_config.get("cortex_llm"),
        }

    @staticmethod
    def _build_mode_section(raw_config: dict) -> Dict:
        """Build the mode-specific fields from a single-mode config.

        Parameters
        ----------
        raw_config : dict
            The original single-mode configuration.

        Returns
        -------
        dict
            Mode-level fields extracted from the single-mode config.
        """
        mode_name = raw_config.get("name", "default")
        return {
            "display_name": mode_name,
            "description": f"Converted from single-mode config '{mode_name}'",
            "hertz": raw_config.get("hertz", 1.0),
            "system_prompt_base": raw_config.get("system_prompt_base", ""),
            "agent_inputs": raw_config.get("agent_inputs", []),
            "agent_actions": raw_config.get("agent_actions", []),
            "backgrounds": raw_config.get("backgrounds", []),
            "simulators": raw_config.get("simulators", []),
            "cortex_llm": raw_config.get("cortex_llm"),
            "action_execution_mode": raw_config.get(
                "action_execution_mode", "concurrent"
            ),
            "action_dependencies": raw_config.get("action_dependencies", {}),
        }

    @staticmethod
    def _validate(converted_config: dict, mode_name: str) -> None:
        """Validate that conversion produced the required multi-mode structure.

        Parameters
        ----------
        converted_config : dict
            The converted multi-mode configuration to validate.
        mode_name : str
            The expected default mode name.

        Raises
        ------
        ValueError
            If required structural fields are missing.
        """
        global_required = ["default_mode", "modes"]
        for key in global_required:
            if key not in converted_config or converted_config[key] is None:
                raise ValueError(
                    f"Conversion failed: missing global required field '{key}'"
                )
        if mode_name not in converted_config["modes"]:
            raise ValueError(
                f"Conversion failed: default_mode '{mode_name}' not in modes"
            )

        mode_required = ["display_name", "description"]
        mode = converted_config["modes"][mode_name]
        for key in mode_required:
            if key not in mode or mode[key] is None:
                raise ValueError(
                    f"Conversion failed: missing required field '{key}' in mode '{mode_name}'"
                )

        logging.info(f"Conversion validated: config '{mode_name}'")


convert_to_multi_mode = ConfigConverter.convert_to_multi_mode
