import importlib
import inspect
import logging
import os
import re
import typing as T
from pathlib import Path
from typing import Optional, Set

from simulators.base import Simulator, SimulatorConfig

_VALID_SIMULATOR_MODULE_NAMES: Optional[Set[str]] = None


def _enumerate_valid_simulator_modules(
    simulators_plugins_dir: str = "src/ssimulators/plugins",
) -> Set[str]:
    """Enumerates valid simulator module names from the simulators plugins directory."""
    valid_modules: Set[str] = set()
    plugins_path = Path(simulators_plugins_dir)

    if not plugins_path.is_dir():
        logging.warning(
            f"Simulator plugins directory '{simulators_plugins_dir}' not found."
        )
        return valid_modules

    for plugin_file in plugins_path.iterdir():
        if plugin_file.suffix == ".py" and plugin_file.name != "__init__.py":
            module_name = plugin_file.stem
            valid_modules.add(module_name)

    return valid_modules


def _get_valid_simulator_module_names() -> Set[str]:
    """Gets the cached set of valid simulator module names."""
    global _VALID_SIMULATOR_MODULE_NAMES
    if _VALID_SIMULATOR_MODULE_NAMES is None:
        _VALID_SIMULATOR_MODULE_NAMES = _enumerate_valid_simulator_modules()
    return _VALID_SIMULATOR_MODULE_NAMES


def _validate_simulator_module_name(module_name: str) -> bool:
    """Validates module_name against the whitelist."""
    valid_modules = _get_valid_simulator_module_names()
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

            pattern = rf"^class\s+{re.escape(class_name)}\s*\([^)]*Simulator[^)]*\)\s*:"

            if re.search(pattern, content, re.MULTILINE):
                candidate_module_name = plugin_file[:-3]
                if not _validate_simulator_module_name(candidate_module_name):
                    logging.warning(
                        f"Potential security issue: Found class '{class_name}' in module '{candidate_module_name}', but module name is not in whitelist. Skipping."
                    )
                    continue
                return candidate_module_name

        except Exception as e:
            logging.warning(f"Could not read {plugin_file}: {e}")
            continue

    return None


def get_simulator_class(class_name: str) -> T.Type[Simulator]:
    """
    Get a Simulator class by its class name.

    Parameters
    ----------
    class_name : str
        The exact class name

    Returns
    -------
    T.Type[Simulator]
        The Simulator class
    """
    module_name = find_module_with_class(class_name)

    if module_name is None:
        raise ValueError(
            f"Class '{class_name}' not found in any simulator plugin module"
        )

    if not re.match(r"^[a-zA-Z0-9_-]+$", module_name):
        raise ValueError(
            f"Invalid characters in simulator module name '{module_name}'. Only alphanumeric, underscore, and hyphen are allowed."
        )

    try:
        module = importlib.import_module(f"simulators.plugins.{module_name}")
        simulator_class = getattr(module, class_name)

        if not (
            inspect.isclass(simulator_class)
            and issubclass(simulator_class, Simulator)
            and simulator_class != Simulator
        ):
            raise ValueError(f"'{class_name}' is not a valid Simulator subclass")

        logging.debug(f"Got Simulator class {class_name} from {module_name}.py")
        return simulator_class

    except ImportError as e:
        raise ValueError(f"Could not import simulator module '{module_name}': {e}")
    except AttributeError:
        raise ValueError(
            f"Class '{class_name}' not found in simulator module '{module_name}'"
        )


def load_simulator(simulator_config: T.Dict[str, T.Any]) -> Simulator:
    """
    Load a Simulator instance with its configuration.

    Parameters
    ----------
    simulator_config : dict
        Configuration dictionary

    Returns
    -------
    Simulator
        The instantiated simulator
    """
    class_name = simulator_config["type"]
    module_name = find_module_with_class(class_name)

    if module_name is None:
        raise ValueError(
            f"Class '{class_name}' not found in any simulator plugin module"
        )

    if not re.match(r"^[a-zA-Z0-9_-]+$", module_name):
        raise ValueError(
            f"Invalid characters in simulator module name '{module_name}'. Only alphanumeric, underscore, and hyphen are allowed."
        )

    try:
        module = importlib.import_module(f"simulators.plugins.{module_name}")
        simulator_class = getattr(module, class_name)

        if not (
            inspect.isclass(simulator_class)
            and issubclass(simulator_class, Simulator)
            and simulator_class != Simulator
        ):
            raise ValueError(f"'{class_name}' is not a valid simulator subclass")

        config_class = None
        for _, obj in module.__dict__.items():
            if (
                isinstance(obj, type)
                and issubclass(obj, SimulatorConfig)
                and obj != SimulatorConfig
            ):
                config_class = obj

        config_dict = simulator_config.get("config", {})
        if config_class is not None:
            config = config_class(
                **(config_dict if isinstance(config_dict, dict) else {})
            )
        else:
            config = SimulatorConfig(
                **(config_dict if isinstance(config_dict, dict) else {})
            )

        logging.debug(f"Loaded simulator {class_name} from {module_name}.py")
        return simulator_class(config=config)

    except ImportError as e:
        raise ValueError(f"Could not import simulator module '{module_name}': {e}")
    except AttributeError:
        raise ValueError(
            f"Class '{class_name}' not found in simulator module '{module_name}'"
        )
