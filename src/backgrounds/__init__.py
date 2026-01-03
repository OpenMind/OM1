import importlib
import inspect
import logging
import os
import re
import typing as T

from backgrounds.base import Background, BackgroundConfig


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
                rf"^class\s+{re.escape(class_name)}\s*\([^)]*Background[^)]*\)\s*:"
            )

            if re.search(pattern, content, re.MULTILINE):
                return plugin_file[:-3]

        except Exception as e:
            logging.warning(f"Could not read {plugin_file}: {e}")
            continue

    return None


def load_background(background_config: T.Dict[str, T.Any]) -> Background:
    """
    Load a background instance with its configuration.

    Parameters
    ----------
    background_config : dict
        Configuration dictionary

    Returns
    -------
    Background
        The instantiated background
    """
    class_name = background_config["type"]
    module_name = find_module_with_class(class_name)

    if module_name is None:
        raise ValueError(
            f"Class '{class_name}' not found in any background plugin module"
        )

    try:
        module = importlib.import_module(f"backgrounds.plugins.{module_name}")
        background_class = getattr(module, class_name)

        if not (
            inspect.isclass(background_class)
            and issubclass(background_class, Background)
            and background_class != Background
        ):
            raise ValueError(f"'{class_name}' is not a valid background subclass")

        config_class = None
        for _, obj in module.__dict__.items():
            if (
                isinstance(obj, type)
                and issubclass(obj, BackgroundConfig)
                and obj != BackgroundConfig
            ):
                config_class = obj

        config_dict = background_config.get("config", {})
        if config_class is not None:
            config = config_class(
                **(config_dict if isinstance(config_dict, dict) else {})
            )
        else:
            config = BackgroundConfig(
                **(config_dict if isinstance(config_dict, dict) else {})
            )

        logging.debug(f"Loaded background {class_name} from {module_name}.py")
        return background_class(config=config)

    except ImportError as e:
        raise ValueError(f"Could not import background module '{module_name}': {e}")
    except AttributeError:
        raise ValueError(
            f"Class '{class_name}' not found in background module '{module_name}'"
        )
