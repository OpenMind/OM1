import importlib
import inspect
import logging
import os
import re
import typing as T
import json
import shutil
from pathlib import Path

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

        # --- HISTORY PERSISTENCE IMPLEMENTATION ---
        # 1. Define history file path
        # History stored in 'data/history_{AGENT_NAME}.json'
        self.history_file = Path("data") / f"history_{config.agent_name}.json"
        
        # 2. Load history from disk on startup
        self.history: T.List[T.Dict[str, str]] = self._load_history()
        # --- END HISTORY PERSISTENCE ---

        # Set up available actions for function calling
        self._available_actions = available_actions or []
        self.function_schemas = []
        if self._available_actions:
            self.function_schemas = generate_function_schemas_from_actions(
                self._available_actions
            )
            logging.info(
                f"LLM initialized with {len(self.function_schemas)} function schemas"
            )

        # Set up the IO provider
        self.io_provider = IOProvider()

    # --- NEW METHODS FOR HISTORY MANAGEMENT ---

    def _load_history(self) -> T.List[T.Dict[str, str]]:
        """
        Safely load conversation history from disk.
        Handles missing files and corrupted JSON gracefully to prevent crashes.
        """
        if not self.history_file.exists():
            logging.info(f"No history found at {self.history_file}, starting fresh.")
            return []

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Enforce history length limit if configured
            limit = self._config.history_length
            if limit and limit > 0 and len(data) > limit:
                data = data[-limit:]
                
            logging.info(f"Restored {len(data)} messages from {self.history_file}")
            return data
        except Exception as e:
            logging.error(f"Failed to load history from {self.history_file}: {e}")
            return []

    def _save_history(self):
        """
        ATOMIC WRITE: Save history to disk safely.
        Writes to a temporary .tmp file first, then renames it.
        This prevents data corruption if the system crashes or loses power during write.
        """
        # Ensure parent directory exists
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        temp_file = self.history_file.with_suffix(".tmp")
        
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            
            # Atomic rename (POSIX compliant) guarantees data integrity
            shutil.move(str(temp_file), str(self.history_file))
            
        except Exception as e:
            logging.error(f"Failed to save history to {self.history_file}: {e}")

    def add_message(self, role: str, content: str):
        """
        Add a message to history and trigger an immediate atomic save.
        
        This method should be used instead of appending to self.history directly
        to ensure data is persisted to disk.
        """
        self.history.append({"role": role, "content": content})
        
        # Prune history if it exceeds the configured limit
        limit = self._config.history_length
        if limit and limit > 0 and len(self.history) > limit:
            self.history = self.history[-limit:]
            
        self._save_history()
        
    def clear_history(self):
        """Clear all conversation history from both memory and disk."""
        self.history = []
        self._save_history()

    # --- END NEW METHODS ---

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

        if not (
            inspect.isclass(llm_class)
            and issubclass(llm_class, LLM)
            and llm_class != LLM
        ):
            raise ValueError(f"'{class_name}' is not a valid LLM subclass")

        logging.debug(f"Loaded LLM {class_name} from {module_name}.py")
        return llm_class

    except ImportError as e:
        raise ValueError(f"Could not import LLM module '{module_name}': {e}")
    except AttributeError:
        raise ValueError(
            f"Class '{class_name}' not found in LLM module '{module_name}'"
        )