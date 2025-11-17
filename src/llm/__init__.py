import importlib
import inspect
import logging
import os
import re
import typing as T

from pydantic import BaseModel, ConfigDict, Field

from llm.function_schemas import generate_function_schemas_from_actions
from providers.io_provider import IOProvider

R = T.TypeVar("R")


class LLMConfig(BaseModel):
    """
    Configuration class for Language Learning Models.

    Parameters
    ----------
    base_url : str, optional
        Base URL for the LLM API endpoint
    api_key : str, optional
        Authentication key for the LLM service
    model : str, optional
        Name of the LLM model to use
    history_length : int, optional
        Number of interactions to store in the history buffer
    extra_params : dict, optional
        Additional parameters for the LLM API request
    """

    model_config = ConfigDict(extra="allow")

    base_url: T.Optional[str] = None
    api_key: T.Optional[str] = None
    model: T.Optional[str] = None
    timeout: T.Optional[int] = 10
    agent_name: T.Optional[str] = "IRIS"
    history_length: T.Optional[int] = 0
    extra_params: T.Dict[str, T.Any] = Field(default_factory=dict)

    def __getitem__(self, item: str) -> T.Any:
        """
        Get an item from the configuration.

        Parameters
        ----------
        item : str
            The key to retrieve from the configuration

        Returns
        -------
        T.Any
            The value associated with the key in the configuration
        """
        try:
            return getattr(self, item)
        except AttributeError:
            return self.extra_params[item]

    def __setitem__(self, key: str, value: T.Any) -> None:
        """
        Set an item in the configuration.

        Parameters
        ----------
        key : str
            The key to set in the configuration
        value : T.Any
            The value to associate with the key in the configuration
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extra_params[key] = value


class LLM(T.Generic[R]):
    """
    Base class for Language Learning Model implementations.

    Generic interface for implementing LLM clients with type-safe responses.

    Parameters
    ----------
    output_model : Type[R]
        Type specification for model responses
    config : LLMConfig, optional
        Configuration settings for the LLM
    available_actions : list, optional
        List of available actions for function calling
    """

    def __init__(
        self,
        config: LLMConfig = LLMConfig(),
        available_actions: T.Optional[list] = None,
    ):
        # Set up the LLM configuration
        self._config = config

        # Set up available actions for function calling
        self._available_actions = available_actions or []
        self.function_schemas = []
        if self._available_actions:
            self.function_schemas = generate_function_schemas_from_actions(self._available_actions)
            logging.info(f"LLM initialized with {len(self.function_schemas)} function schemas")

        # Set up the IO provider
        self.io_provider = IOProvider()

    async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]] = []) -> R:
        """
        Send a prompt to the LLM and receive a typed response.

        Parameters
        ----------
        prompt : str
            Input text to send to the model
        messages : List[Dict[str, str]]
            List of message dictionaries to send to the model.

        Returns
        -------
        R
            Response matching the output_model type specification

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses
        """
        raise NotImplementedError


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

            pattern = rf"^class\s+{re.escape(class_name)}\s*\([^)]*LLM[^)]*\)\s*:"

            if re.search(pattern, content, re.MULTILINE):
                return plugin_file[:-3]

        except Exception as e:
            logging.warning(f"Could not read {plugin_file}: {e}")
            continue

    return None


def load_llm(class_name: str) -> T.Type[LLM]:
    """
    Load an LLM class by its class name.

    Parameters
    ----------
    class_name : str
        The exact class name

    Returns
    -------
    T.Type[LLM]
        The LLM class
    """
    module_name = find_module_with_class(class_name)

    if module_name is None:
        raise ValueError(f"Class '{class_name}' not found in any LLM plugin module")

    try:
        module = importlib.import_module(f"llm.plugins.{module_name}")
        llm_class = getattr(module, class_name)

        if not (inspect.isclass(llm_class) and issubclass(llm_class, LLM) and llm_class != LLM):
            raise ValueError(f"'{class_name}' is not a valid LLM subclass")

        logging.debug(f"Loaded LLM {class_name} from {module_name}.py")
        return llm_class

    except ImportError as e:
        raise ValueError(f"Could not import LLM module '{module_name}': {e}")
    except AttributeError:
        raise ValueError(f"Class '{class_name}' not found in LLM module '{module_name}'")
