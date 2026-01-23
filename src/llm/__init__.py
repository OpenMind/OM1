import asyncio
import importlib
import inspect
import logging
import os
import re
import typing as T
from functools import wraps

from pydantic import BaseModel, ConfigDict, Field

from llm.function_schemas import generate_function_schemas_from_actions
from providers.io_provider import IOProvider

R = T.TypeVar("R")


def with_llm_retry(max_retries: T.Optional[int] = None, backoff_base: T.Optional[float] = None):
    """
    Decorator to add retry logic to LLM API calls with exponential backoff.

    This decorator wraps async LLM methods to automatically retry on failures,
    improving reliability when dealing with temporary network issues, rate limits,
    or transient API errors.

    Parameters
    ----------
    max_retries : int, optional
        Maximum number of retry attempts. If None, uses config value (default: 3)
    backoff_base : float, optional
        Base for exponential backoff calculation in seconds.
        If None, uses config value (default: 1.5)
        Wait time = backoff_base ** attempt (e.g., 1s, 1.5s, 2.25s)

    Returns
    -------
    Callable
        Decorated async function with retry logic

    Examples
    --------
    >>> # Use default values from config
    >>> @with_llm_retry()
    >>> async def ask(self, prompt, messages):
    >>>     return await self._client.chat.completions.create(...)
    
    >>> # Override for time-sensitive applications
    >>> @with_llm_retry(max_retries=2, backoff_base=1.0)
    >>> async def ask(self, prompt, messages):
    >>>     return await self._client.chat.completions.create(...)

    Notes
    -----
    - Retries on any Exception
    - Uses exponential backoff to avoid overwhelming the API
    - Logs warnings on retry attempts and errors on final failure
    - Returns None after all retries are exhausted
    - Can be configured via LLMConfig or decorator parameters
    """

    def decorator(func: T.Callable[..., T.Awaitable[T.Optional[R]]]) -> T.Callable[..., T.Awaitable[T.Optional[R]]]:
        @wraps(func)
        async def wrapper(self: T.Any, *args: T.Any, **kwargs: T.Any) -> T.Optional[R]:
            # Get retry config from instance or use decorator params
            retries = max_retries if max_retries is not None else getattr(self, '_max_retries', 3)
            backoff = backoff_base if backoff_base is not None else getattr(self, '_retry_backoff_base', 1.5)
            
            for attempt in range(retries):
                try:
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    if attempt < retries - 1:
                        wait_time = backoff ** attempt
                        logging.warning(
                            f"LLM call failed (attempt {attempt + 1}/{retries}), "
                            f"retrying in {wait_time:.1f}s: {type(e).__name__}: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logging.error(
                            f"LLM call failed after {retries} attempts: "
                            f"{type(e).__name__}: {e}"
                        )
                        logging.error(
                            "=" * 70
                        )
                        logging.error(
                            "LLM is unavailable. The system will continue attempting to call "
                            "the LLM in the next cycle."
                        )
                        logging.error(
                            "Options:"
                        )
                        logging.error(
                            "  1. Wait - The system will automatically retry in the next cycle"
                        )
                        logging.error(
                            "  2. Exit - Press Ctrl+C to stop the process"
                        )
                        logging.error(
                            "=" * 70
                        )
                        # Return None to allow the loop to continue
                        # The runtime will skip this tick and try again in the next cycle
                        return None

        return wrapper

    return decorator


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

    base_url: T.Optional[str] = Field(
        default=None, description="Base URL for the LLM API endpoint"
    )
    api_key: T.Optional[str] = Field(
        default=None, description="Authentication key for the LLM service"
    )
    model: T.Optional[str] = Field(
        default=None, description="Name of the LLM model to use"
    )
    timeout: T.Optional[int] = Field(
        default=10, description="Request timeout in seconds"
    )
    agent_name: T.Optional[str] = Field(
        default="IRIS", description="Name of the agent identity"
    )
    history_length: T.Optional[int] = Field(
        default=0, description="Number of past interactions to keep in context"
    )
    max_retries: T.Optional[int] = Field(
        default=3, description="Maximum number of retry attempts for LLM calls"
    )
    retry_backoff_base: T.Optional[float] = Field(
        default=1.5, description="Base for exponential backoff in seconds (e.g., 1.5 means 1s, 1.5s, 2.25s)"
    )
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
        config: LLMConfig,
        available_actions: T.Optional[list] = None,
    ):
        # Set up the LLM configuration
        self._config = config

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

        # Enable state management by default
        self._skip_state_management: bool = False
        
        # Get retry configuration from config
        self._max_retries = getattr(config, 'max_retries', 3)
        self._retry_backoff_base = getattr(config, 'retry_backoff_base', 2.0)

    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, str]] = []
    ) -> T.Optional[R]:
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


def get_llm_class(class_name: str) -> T.Type[LLM]:
    """
    Get an LLM class by its class name.

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

        logging.debug(f"Got LLM class {class_name} from {module_name}.py")
        return llm_class

    except ImportError as e:
        raise ValueError(f"Could not import LLM module '{module_name}': {e}")
    except AttributeError:
        raise ValueError(
            f"Class '{class_name}' not found in LLM module '{module_name}'"
        )


def load_llm(
    llm_config: T.Dict[str, T.Any],
    available_actions: T.Optional[list] = None,
) -> LLM:
    """
    Load an LLM instance with its configuration.

    Parameters
    ----------
    llm_config : dict
        Configuration dictionary
    available_actions : list, optional
        List of available actions for function calling

    Returns
    -------
    LLM
        The instantiated LLM
    """
    class_name = llm_config["type"]
    module_name = find_module_with_class(class_name)

    if module_name is None:
        raise ValueError(f"Class '{class_name}' not found in LLM plugin module")

    try:
        module = importlib.import_module(f"llm.plugins.{module_name}")
        llm_class = getattr(module, class_name)

        if not (
            inspect.isclass(llm_class)
            and issubclass(llm_class, LLM)
            and llm_class != LLM
        ):
            raise ValueError(f"'{class_name}' is not a valid LLM subclass")

        config_class = None
        for _, obj in module.__dict__.items():
            if (
                isinstance(obj, type)
                and issubclass(obj, LLMConfig)
                and obj != LLMConfig
            ):
                config_class = obj

        config_dict = llm_config.get("config", {})
        if config_class is not None:
            config = config_class(
                **(config_dict if isinstance(config_dict, dict) else {})
            )
        else:
            config = LLMConfig(**(config_dict if isinstance(config_dict, dict) else {}))

        logging.debug(f"Loaded LLM {class_name} from {module_name}.py")
        return llm_class(config=config, available_actions=available_actions)

    except ImportError as e:
        raise ValueError(f"Could not import LLM module '{module_name}': {e}")
    except AttributeError:
        raise ValueError(
            f"Class '{class_name}' not found in LLM module '{module_name}'"
        )
