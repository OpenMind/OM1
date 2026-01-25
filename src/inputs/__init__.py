import importlib
import inspect
import logging
import os
import re
import typing as T
from pathlib import Path
from typing import Optional, Set

from inputs.base import Sensor, SensorConfig

_VALID_INPUT_MODULE_NAMES: Optional[Set[str]] = None


def _enumerate_valid_input_modules(
    inputs_plugins_dir: str = "src/inputs/plugins",
) -> Set[str]:
    """Enumerates valid input module names from the inputs plugins directory."""
    valid_modules: Set[str] = set()
    plugins_path = Path(inputs_plugins_dir)

    if not plugins_path.is_dir():
        logging.warning(f"Input plugins directory '{inputs_plugins_dir}' not found.")
        return valid_modules

    for plugin_file in plugins_path.iterdir():
        if plugin_file.suffix == ".py" and plugin_file.name != "__init__.py":
            module_name = plugin_file.stem
            valid_modules.add(module_name)

    return valid_modules


def _get_valid_input_module_names() -> Set[str]:
    """Gets the cached set of valid input module names."""
    global _VALID_INPUT_MODULE_NAMES
    if _VALID_INPUT_MODULE_NAMES is None:
        _VALID_INPUT_MODULE_NAMES = _enumerate_valid_input_modules()
    return _VALID_INPUT_MODULE_NAMES


def _validate_input_module_name(module_name: str) -> bool:
    """Validates module_name against the whitelist."""
    valid_modules = _get_valid_input_module_names()
    return module_name in valid_modules


def find_module_with_class(class_name: str) -> T.Optional[str]:
    """
    Find which module file contains the specified class name.

    Parameters
    ----------
    class_name : str
        The class name to search for

    Returns
    -------
    str or None
        The module name (without .py) that contains the class, or None if not found
    """
    plugins_dir = os.path.join(os.path.dirname(__file__), "plugins")

    if not os.path.exists(plugins_dir):
        return None

    plugin_files = [f for f in os.listdir(plugins_dir) if f.endswith(".py")]

    for plugin_file in plugin_files:
        file_path = os.path.join(plugins_dir, plugin_file)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            pattern = (
                rf"^class\s+{re.escape(class_name)}\s*\([^)]*FuserInput[^)]*\)\s*:"
            )

            if re.search(pattern, content, re.MULTILINE):
                candidate_module_name = plugin_file[:-3]
                if not _validate_input_module_name(candidate_module_name):
                    logging.warning(
                        f"Potential security issue: Found class '{class_name}' in module '{candidate_module_name}', but module name is not in whitelist. Skipping."
                    )
                    continue
                return candidate_module_name

        except Exception as e:
            logging.warning(f"Could not read {plugin_file}: {e}")
            continue

    return None


def load_input(input_config: T.Dict[str, T.Any]) -> Sensor:
    """
    Load an input and configuration.

    Parameters
    ----------
    input_config : dict

    Returns
    -------
    Sensor
        The instantiated sensor
    """
    class_name = input_config["type"]
    module_name = find_module_with_class(class_name)

    if module_name is None:
        raise ValueError(f"Class '{class_name}' not found in any input plugin module")

    if not re.match(r"^[a-zA-Z0-9_-]+$", module_name):
        raise ValueError(
            f"Invalid characters in input module name '{module_name}'. Only alphanumeric, underscore, and hyphen are allowed."
        )

    try:
        module = importlib.import_module(f"inputs.plugins.{module_name}")
        input_class = getattr(module, class_name)

        if not (
            inspect.isclass(input_class)
            and issubclass(input_class, Sensor)
            and input_class != Sensor
        ):
            raise ValueError(f"'{class_name}' is not a valid input subclass")

        config_class = None
        for _, obj in module.__dict__.items():
            if (
                isinstance(obj, type)
                and issubclass(obj, SensorConfig)
                and obj != SensorConfig
            ):
                config_class = obj

        config_dict = input_config.get("config", {})
        if config_class is not None:
            config = config_class(
                **(config_dict if isinstance(config_dict, dict) else {})
            )
        else:
            config = SensorConfig(
                **(config_dict if isinstance(config_dict, dict) else {})
            )

        logging.debug(f"Loaded input {class_name} from {module_name}.py")
        return input_class(config=config)

    except ImportError as e:
        raise ValueError(f"Could not import input module '{module_name}': {e}")
    except AttributeError:
        raise ValueError(
            f"Class '{class_name}' not found in input module '{module_name}'"
        )
