import logging


def is_single_mode(raw_config: dict) -> bool:
    """Detect whether the configuration is in single-mode format."""
    return "modes" not in raw_config or "default_mode" not in raw_config


def normalize_to_multi_mode(raw_config: dict) -> dict:
    """
    Normalize a single-mode config to multi-mode format.

    If the config is already multi-mode, return it unchanged.
    """
    if not is_single_mode(raw_config):
        return raw_config

    mode_name = raw_config.get("name", "default")
    logging.info(f"Normalizing single-mode config '{mode_name}'")

    normalized_config = _build_global_section(raw_config, mode_name)
    normalized_config["modes"] = {mode_name: _build_mode_section(raw_config)}
    normalized_config["transition_rules"] = []

    _validate_normalized(normalized_config, mode_name)

    return normalized_config


def _build_global_section(raw_config: dict, mode_name: str) -> dict:
    """Build the global fields of a multi-mode config."""
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


def _build_mode_section(raw_config: dict) -> dict:
    """Build the mode-specific fields from a single-mode config."""
    mode_name = raw_config.get("name", "default")
    return {
        "display_name": mode_name,
        "description": f"Normalized config from single-mode config '{mode_name}'",
        "hertz": raw_config.get("hertz", 1.0),
        "system_prompt_base": raw_config.get("system_prompt_base", ""),
        "agent_inputs": raw_config.get("agent_inputs", []),
        "agent_actions": raw_config.get("agent_actions", []),
        "backgrounds": raw_config.get("backgrounds", []),
        "simulators": raw_config.get("simulators", []),
        "cortex_llm": raw_config.get("cortex_llm"),
        "action_execution_mode": raw_config.get("action_execution_mode", "concurrent"),
        "action_dependencies": raw_config.get("action_dependencies", {}),
    }


def _validate_normalized(normalized_config: dict, mode_name: str) -> None:
    """Validate that normalization is correct."""
    global_required = [
        "version",
        "default_mode",
        "api_key",
        "system_governance",
        "cortex_llm",
        "modes",
    ]
    for field in global_required:
        if field not in normalized_config or normalized_config[field] is None:
            raise ValueError(
                f"Normalization failed: missing global required field '{field}'"
            )
    if mode_name not in normalized_config["modes"]:
        raise ValueError(
            f"Normalization failed: default_mode '{mode_name}' not in modes"
        )

    mode_required = [
        "display_name",
        "description",
        "system_prompt_base",
        "hertz",
        "agent_inputs",
        "agent_actions",
    ]
    mode = normalized_config["modes"][mode_name]
    for field in mode_required:
        if field not in mode or mode[field] is None:
            raise ValueError(
                f"Normalization failed: missing required field '{field}' in mode '{mode_name}'"
            )

    logging.info(f"Normalization validated: config '{mode_name}'")
