import importlib
import inspect
import logging
import os
import re
import typing as T

from inputs.base import Sensor, SensorConfig


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
                return plugin_file[:-3]

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
