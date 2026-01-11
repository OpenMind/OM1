"""Configuration file utilities for CLI."""

import os
from pathlib import Path
from typing import List, Tuple

import json5


def get_config_dir() -> Path:
    """
    Get the configuration directory path.

    Returns
    -------
    Path
        Path to the config directory
    """
    return Path(__file__).parent.parent.parent.parent / "config"


def get_src_dir() -> Path:
    """
    Get the source directory path.

    Returns
    -------
    Path
        Path to the src directory
    """
    return Path(__file__).parent.parent.parent


def resolve_config_path(config_name: str) -> Path:
    """
    Resolve configuration path from name or path.

    Parameters
    ----------
    config_name : str
        Configuration name or path

    Returns
    -------
    Path
        Absolute path to configuration file

    Raises
    ------
    FileNotFoundError
        If configuration file cannot be found
    """
    if os.path.exists(config_name):
        return Path(config_name).resolve()

    if os.path.exists(config_name + ".json5"):
        return Path(config_name + ".json5").resolve()

    config_dir = get_config_dir()
    config_path = config_dir / config_name

    if config_path.exists():
        return config_path.resolve()

    if (config_path.parent / (config_path.name + ".json5")).exists():
        return (config_path.parent / (config_path.name + ".json5")).resolve()

    config_with_ext = config_dir / (config_name + ".json5")
    if config_with_ext.exists():
        return config_with_ext.resolve()

    raise FileNotFoundError(
        f"Configuration '{config_name}' not found. "
        f"Tried: {config_name}, {config_name}.json5, "
        f"{config_path}, {config_with_ext}"
    )


def list_configs() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    List all available configuration files.

    Returns
    -------
    Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]
        Tuple of (standard_configs, mode_aware_configs)
        Each config is a tuple of (filename, display_name)
    """
    config_dir = get_config_dir()

    if not config_dir.exists():
        return [], []

    standard_configs = []
    mode_configs = []

    for config_file in sorted(config_dir.glob("*.json5")):
        config_name = config_file.stem

        try:
            with open(config_file, "r") as f:
                raw_config = json5.load(f)

            display_name = raw_config.get("name", config_name)

            if "modes" in raw_config and "default_mode" in raw_config:
                mode_configs.append((config_name, display_name))
            else:
                standard_configs.append((config_name, display_name))
        except Exception:
            standard_configs.append((config_name, "Invalid config"))

    return standard_configs, mode_configs


def is_mode_aware_config(config_path: Path) -> bool:
    """
    Check if a configuration is mode-aware.

    Parameters
    ----------
    config_path : Path
        Path to configuration file

    Returns
    -------
    bool
        True if configuration is mode-aware
    """
    try:
        with open(config_path, "r") as f:
            raw_config = json5.load(f)
        return "modes" in raw_config and "default_mode" in raw_config
    except Exception:
        return False


def load_config_raw(config_path: Path) -> dict:
    """
    Load raw configuration without processing.

    Parameters
    ----------
    config_path : Path
        Path to configuration file

    Returns
    -------
    dict
        Raw configuration dictionary
    """
    with open(config_path, "r") as f:
        return json5.load(f)


def complete_config_name(incomplete: str) -> List[str]:
    """
    Auto-complete config names for CLI.

    Parameters
    ----------
    incomplete : str
        Partial config name

    Returns
    -------
    List[str]
        List of matching config names
    """
    config_dir = get_config_dir()
    return [
        f.stem
        for f in config_dir.glob("*.json5")
        if f.stem.startswith(incomplete) and not f.stem.startswith(".")
    ]
