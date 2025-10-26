import importlib
import inspect
import logging
import os
import re
import typing as T
from inputs.base import Sensor


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


def _create_stub_sensor(class_name: str, module_name: str, error: Exception) -> T.Type[Sensor]:
    """
    Create a stub Sensor class for when optional dependencies are missing.

    Parameters
    ----------
    class_name : str
        The name of the class that failed to load
    module_name : str
        The name of the module that failed to import
    error : Exception
        The import error that occurred

    Returns
    -------
    T.Type[Sensor]
        A stub sensor class that warns when instantiated
    """
    class StubSensor(Sensor):
        def __init__(self, *args, **kwargs):
            logging.warning(
                f"Stub sensor '{class_name}' instantiated. "
                f"Original module '{module_name}' failed to load: {error}\n"
                f"This sensor will not provide any data. "
                f"Install missing dependencies to enable this sensor."
            )
            super().__init__(*args, **kwargs)

        def read(self):
            """Return empty/no-op data."""
            return None

    StubSensor.__name__ = f"Stub_{class_name}"
    StubSensor.__qualname__ = f"Stub_{class_name}"
    return StubSensor


def load_input(class_name: str) -> T.Type[Sensor]:
    """
    Load an input class by its class name.

    This function attempts to dynamically load sensor plugins. If a plugin's
    dependencies are missing, it returns a stub sensor that logs warnings
    instead of crashing the application.

    Parameters
    ----------
    class_name : str
        The exact class name

    Returns
    -------
    T.Type[Sensor]
        The sensor class, or a stub sensor if dependencies are missing
    """
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

        logging.debug(f"Loaded input {class_name} from {module_name}.py")
        return input_class

    except ImportError as e:
        # Handle missing optional dependencies gracefully
        logging.warning(
            f"Optional input module '{module_name}' could not be imported: {e}\n"
            f"This is likely due to missing dependencies. "
            f"A stub sensor will be registered for '{class_name}'.\n"
            f"To enable this sensor, install the required dependencies."
        )
        return _create_stub_sensor(class_name, module_name, e)

    except AttributeError:
        raise ValueError(
            f"Class '{class_name}' not found in input module '{module_name}'"
        )
