import asyncio
import importlib
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.kokoro_tts_provider import KokoroTTSProvider
from providers.riva_tts_provider import RivaTTSProvider


class LifecycleHookType(Enum):
    """
    Types of lifecycle hooks.

    - ON_ENTRY: Execute when entering the mode
    - ON_EXIT: Execute when exiting the mode
    - ON_STARTUP: Execute when the mode system first starts
    - ON_SHUTDOWN: Execute when the mode system shuts down
    - ON_TIMEOUT: Execute when the mode times out
    """

    ON_ENTRY = "on_entry"
    ON_EXIT = "on_exit"
    ON_STARTUP = "on_startup"
    ON_SHUTDOWN = "on_shutdown"
    ON_TIMEOUT = "on_timeout"


@dataclass
class LifecycleHook:
    """
    Configuration for a lifecycle hook.

    Parameters
    ----------
    hook_type : LifecycleHookType
        The type of lifecycle hook
    handler_type : str
        The type of handler ('action', 'function', 'message', 'command')
    handler_config : Dict
        Configuration for the handler
    async_execution : bool
        Whether to execute the hook asynchronously (default: True)
    timeout_seconds : Optional[float]
        Timeout for hook execution (default: 5.0 seconds)
    on_failure : str
        Action to take on failure ('ignore', 'abort') (default: 'ignore')
    priority : int
        Execution priority for multiple hooks of same type (higher = first) (default: 0)
    """

    hook_type: LifecycleHookType
    handler_type: str
    handler_config: Dict[str, Any]
    async_execution: bool = True
    timeout_seconds: Optional[float] = 5.0
    on_failure: str = "ignore"
    priority: int = 0


class HookConfig(BaseModel):
    """
    Base configuration class for hook handlers.
    """

    model_config = ConfigDict(extra="allow")


class MessageHookConfig(HookConfig):
    """
    Configuration for MessageHookHandler.

    Parameters
    ----------
    message : str
        The message to log or announce. Supports {variable} formatting.
    tts_provider : str
        The TTS provider to use ('elevenlabs', 'kokoro', 'riva'). Defaults to 'elevenlabs'.
    base_url : Optional[str]
        The URL endpoint for the TTS service. Provider-specific defaults apply.
    api_key : Optional[str]
        OpenMind API key for TTS service.
    elevenlabs_api_key : Optional[str]
        ElevenLabs API key (only for 'elevenlabs' provider).
    voice_id : str
        Voice ID for the TTS provider.
    model_id : str
        Model ID for the TTS provider.
    output_format : str
        Audio output format.
    rate : Optional[int]
        Audio sample rate in Hz (only for 'kokoro' provider).
    enable_tts_interrupt : bool
        Enable TTS interrupt capability.
    """

    message: str = Field(
        default="",
        description="The message to log or announce. Supports {variable} formatting.",
    )
    tts_provider: str = Field(
        default="elevenlabs",
        description="The TTS provider to use ('elevenlabs', 'kokoro', 'riva')",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="The URL endpoint for the TTS service. Provider-specific defaults apply.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="OpenMind API key for TTS service",
    )
    elevenlabs_api_key: Optional[str] = Field(
        default=None,
        description="ElevenLabs API key (only for 'elevenlabs' provider)",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Voice ID for the TTS provider",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Model ID for the TTS provider",
    )
    output_format: Optional[str] = Field(
        default=None,
        description="Audio output format",
    )
    rate: Optional[int] = Field(
        default=None,
        description="Audio sample rate in Hz (only for 'kokoro' provider)",
    )
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt capability",
    )


class CommandHookConfig(HookConfig):
    """
    Configuration for CommandHookHandler.

    Parameters
    ----------
    command : str
        The shell command to execute. Supports {variable} formatting.
    """

    command: str = Field(
        default="",
        description="The shell command to execute. Supports {variable} formatting.",
    )


class FunctionHookConfig(HookConfig):
    """
    Configuration for FunctionHookHandler.

    Parameters
    ----------
    module_name : str
        Name of the module file (without .py extension) in the hooks directory.
    function : str
        Name of the function to call.
    """

    module_name: str = Field(
        description="Name of the module file (without .py extension) in the hooks directory",
    )
    function: str = Field(
        description="Name of the function to call",
    )


class ActionHookConfig(HookConfig):
    """
    Configuration for ActionHookHandler.

    Parameters
    ----------
    action_type : str
        The type/name of the action to execute.
    action_config : Dict[str, Any]
        Configuration dictionary for the action.
    """

    action_type: str = Field(
        description="The type/name of the action to execute",
    )
    action_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration dictionary for the action",
    )


class LifecycleHookHandler:
    """
    Base class for lifecycle hook handlers.
    """

    def __init__(self, config: HookConfig):
        """
        Initialize the LifecycleHookHandler with configuration.

        Parameters
        ----------
        config : HookConfig
            Configuration object for the hook handler.
        """
        self.config = config

    async def execute(self, context: Dict[str, Any]) -> bool:
        """
        Execute the lifecycle hook.

        Parameters
        ----------
        context : Dict[str, Any]
            Context information for the hook execution

        Returns
        -------
        bool
            True if execution was successful, False otherwise
        """
        raise NotImplementedError


class MessageHookHandler(LifecycleHookHandler):
    """
    Handler that logs or announces a message.
    """

    def __init__(self, config: MessageHookConfig):
        super().__init__(config)
        self.config: MessageHookConfig = config

    async def execute(self, context: Dict[str, Any]) -> bool:
        """
        Execute the lifecycle message.

        Parameters
        ----------
        context : Dict[str, Any]
            Context information for the hook execution

        Returns
        -------
        bool
            True if execution was successful, False otherwise
        """
        if self.config.message:
            try:
                formatted_message = self.config.message.format(**context)
                logging.info(f"Lifecycle hook message: {formatted_message}")

                try:
                    provider = self._create_tts_provider()
                    provider.start()
                    provider.add_pending_message(formatted_message)
                except Exception as e:
                    logging.error(f"Error adding TTS message: {e}")
                    return False

                return True
            except Exception as e:
                logging.error(f"Error formatting lifecycle message: {e}")
                return False
        return True

    def _create_tts_provider(self):
        """
        Create the appropriate TTS provider based on configuration.

        Returns
        -------
        Union[ElevenLabsTTSProvider, KokoroTTSProvider, RivaTTSProvider]
            The configured TTS provider instance

        Raises
        ------
        ValueError
            If an unsupported TTS provider is specified
        """
        provider_type = self.config.tts_provider.lower()

        if provider_type == "elevenlabs":
            return ElevenLabsTTSProvider(
                url=self.config.base_url
                or "https://api.openmind.org/api/core/elevenlabs/tts",
                api_key=self.config.api_key,
                elevenlabs_api_key=self.config.elevenlabs_api_key,
                voice_id=self.config.voice_id or "JBFqnCBsd6RMkjVDRZzb",
                model_id=self.config.model_id or "eleven_flash_v2_5",
                output_format=self.config.output_format or "pcm_16000",
                enable_tts_interrupt=self.config.enable_tts_interrupt,
            )
        elif provider_type == "kokoro":
            return KokoroTTSProvider(
                url=self.config.base_url or "http://127.0.0.1:8880/v1",
                api_key=self.config.api_key,
                voice_id=self.config.voice_id or "af_bella",
                model_id=self.config.model_id or "kokoro",
                output_format=self.config.output_format or "pcm",
                rate=self.config.rate or 24000,
                enable_tts_interrupt=self.config.enable_tts_interrupt,
            )
        elif provider_type == "riva":
            return RivaTTSProvider(
                url=self.config.base_url or "http://127.0.0.1:50051",
                api_key=self.config.api_key,
            )
        else:
            raise ValueError(
                f"Unsupported TTS provider: {provider_type}. "
                f"Supported providers are: elevenlabs, kokoro, riva"
            )


class CommandHookHandler(LifecycleHookHandler):
    """
    Handler that executes a shell command.
    """

    def __init__(self, config: CommandHookConfig):
        super().__init__(config)
        self.config: CommandHookConfig = config

    async def execute(self, context: Dict[str, Any]) -> bool:
        """
        Execute the lifecycle command.

        Parameters
        ----------
        context : Dict[str, Any]
            Context information for the hook execution

        Returns
        -------
        bool
            True if execution was successful, False otherwise
        """
        if not self.config.command:
            logging.warning("No command specified for command hook")
            return False

        try:
            formatted_command = self.config.command.format(**context)

            process = await asyncio.create_subprocess_shell(
                formatted_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                if stdout:
                    logging.info(f"Hook command output: {stdout.decode().strip()}")
                return True
            else:
                logging.error(
                    f"Hook command failed with code {process.returncode}: {stderr.decode().strip()}"
                )
                return False

        except Exception as e:
            logging.error(f"Error executing lifecycle command: {e}")
            return False


class FunctionHookHandler(LifecycleHookHandler):
    """
    Handler that calls a Python function from a specified module.
    """

    def __init__(self, config: FunctionHookConfig):
        super().__init__(config)
        self.config: FunctionHookConfig = config

    async def execute(self, context: Dict[str, Any]) -> bool:
        """
        Execute the lifecycle function.

        Parameters
        ----------
        context : Dict[str, Any]
            Context information for the hook execution

        Returns
        -------
        bool
            True if execution was successful, False otherwise
        """
        try:
            func = self._find_function_in_module(
                self.config.module_name, self.config.function
            )
            if not func:
                return False

            if asyncio.iscoroutinefunction(func):
                result = await func(context)
            else:
                result = func(context)

            return result is not False

        except Exception as e:
            logging.error(f"Error executing lifecycle function: {e}")
            return False

    def _find_function_in_module(self, module_name: str, function_name: str):
        """
        Search for a function in the specified module file using regex.

        Parameters
        ----------
        module_name : str
            Name of the module file (without .py extension)
        function_name : str
            Name of the function to find

        Returns
        -------
        callable or None
            The function if found, None otherwise
        """
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            hooks_dir = os.path.join(current_dir, "..", "..", "hooks")
            hooks_dir = os.path.abspath(hooks_dir)

            if not os.path.exists(hooks_dir):
                logging.error(f"Hooks directory not found at {hooks_dir}")
                return None

            module_file = os.path.join(hooks_dir, f"{module_name}.py")
            if not os.path.exists(module_file):
                logging.error(
                    f"Module file {module_name}.py not found in hooks directory"
                )
                return None

            try:
                with open(module_file, "r", encoding="utf-8") as f:
                    file_content = f.read()

                function_pattern = re.compile(
                    rf"^(?:async\s+)?def\s+{re.escape(function_name)}\s*\(",
                    re.MULTILINE,
                )

                if not function_pattern.search(file_content):
                    logging.error(
                        f"Function {function_name} not found in {module_name}.py"
                    )
                    return None

                try:
                    module = importlib.import_module(f"hooks.{module_name}")
                    if hasattr(module, function_name):
                        func = getattr(module, function_name)
                        logging.debug(
                            f"Successfully loaded function {function_name} from hooks.{module_name}"
                        )
                        return func
                    else:
                        logging.error(
                            f"Function {function_name} found in file but not importable from hooks.{module_name}"
                        )
                        return None

                except ImportError as e:
                    logging.error(f"Failed to import hooks.{module_name}: {e}")
                    return None

            except (IOError, OSError) as e:
                logging.error(f"Failed to read {module_file}: {e}")
                return None

        except Exception as e:
            logging.error(
                f"Error searching for function {function_name} in module {module_name}: {e}"
            )
            return None


class ActionHookHandler(LifecycleHookHandler):
    """
    Handler that executes an agent action.
    """

    def __init__(self, config: ActionHookConfig):
        super().__init__(config)
        self.config: ActionHookConfig = config
        self.action = None

    async def execute(self, context: Dict[str, Any]) -> bool:
        """
        Execute the lifecycle action.

        Parameters
        ----------
        context : Dict[str, Any]
            Context information for the hook execution

        Returns
        -------
        bool
            True if execution was successful, False otherwise
        """
        if not self.action:
            try:
                from actions import load_action

                self.action = load_action(
                    {
                        "type": self.config.action_type,
                        "config": self.config.action_config,
                    }
                )
            except Exception as e:
                logging.error(f"Error loading action for lifecycle hook: {e}")
                return False

        try:
            await self.action.connector.connect(context.get("input_data"))
            return True
        except Exception as e:
            logging.error(f"Error executing lifecycle action: {e}")
            return False


def create_hook_handler(hook: LifecycleHook) -> Optional[LifecycleHookHandler]:
    """
    Create a hook handler instance based on the hook configuration.

    Parameters
    ----------
    hook : LifecycleHook
        The lifecycle hook configuration

    Returns
    -------
    Optional[LifecycleHookHandler]
        The created handler instance or None if creation failed
    """
    handler_type = hook.handler_type.lower()

    try:
        if handler_type == "message":
            config = MessageHookConfig(**hook.handler_config)
            return MessageHookHandler(config)
        elif handler_type == "command":
            config = CommandHookConfig(**hook.handler_config)
            return CommandHookHandler(config)
        elif handler_type == "function":
            config = FunctionHookConfig(**hook.handler_config)
            return FunctionHookHandler(config)
        elif handler_type == "action":
            config = ActionHookConfig(**hook.handler_config)
            return ActionHookHandler(config)
        else:
            logging.error(f"Unknown hook handler type: {handler_type}")
            return None
    except Exception as e:
        logging.error(f"Error creating hook handler config for {handler_type}: {e}")
        return None


def parse_lifecycle_hooks(
    raw_hooks: List[Dict], api_key: Optional[str] = None
) -> List[LifecycleHook]:
    """
    Parse raw lifecycle hooks configuration into LifecycleHook objects.

    Parameters
    ----------
    raw_hooks : List[Dict]
        Raw hook configuration data
    api_key : Optional[str]
        Global API key to inject into message hooks if not specified

    Returns
    -------
    List[LifecycleHook]
        Parsed lifecycle hook objects
    """
    hooks = []
    for hook_data in raw_hooks:
        try:
            handler_config = hook_data.get("handler_config", {}).copy()

            if api_key is not None and "api_key" not in handler_config:
                handler_config["api_key"] = api_key

            hook = LifecycleHook(
                hook_type=LifecycleHookType(hook_data["hook_type"]),
                handler_type=hook_data["handler_type"],
                handler_config=handler_config,
                async_execution=hook_data.get("async_execution", True),
                timeout_seconds=hook_data.get("timeout_seconds", 5.0),
                on_failure=hook_data.get("on_failure", "ignore"),
                priority=hook_data.get("priority", 0),
            )
            hooks.append(hook)
        except (KeyError, ValueError) as e:
            logging.error(f"Error parsing lifecycle hook: {e}")

    return hooks


async def execute_lifecycle_hooks(
    hooks: List[LifecycleHook],
    hook_type: LifecycleHookType,
    context: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Execute all lifecycle hooks of the specified type.

    Parameters
    ----------
    hooks : List[LifecycleHook]
        List of hooks to potentially execute
    hook_type : LifecycleHookType
        The type of lifecycle hooks to execute
    context : Optional[Dict[str, Any]]
        Context information to pass to the hooks

    Returns
    -------
    bool
        True if all hooks executed successfully, False if any failed
    """
    if context is None:
        context = {}

    context.update({"hook_type": hook_type.value})

    relevant_hooks = [hook for hook in hooks if hook.hook_type == hook_type]
    relevant_hooks.sort(key=lambda h: h.priority, reverse=True)

    if not relevant_hooks:
        return True

    logging.info(f"Executing {len(relevant_hooks)} {hook_type.value} hooks")

    all_successful = True

    for hook in relevant_hooks:
        try:
            handler = create_hook_handler(hook)
            if handler:
                if hook.async_execution:
                    if hook.timeout_seconds:
                        success = await asyncio.wait_for(
                            handler.execute(context), timeout=hook.timeout_seconds
                        )
                    else:
                        success = await handler.execute(context)
                else:
                    success = await handler.execute(context)

                if not success:
                    all_successful = False
                    if hook.on_failure == "abort":
                        logging.error(
                            "Lifecycle hook failed with abort policy, stopping execution"
                        )
                        return False
                    if hook.on_failure == "ignore":
                        pass
            else:
                logging.error(
                    f"Failed to create handler for lifecycle hook: {hook.handler_type}"
                )
                all_successful = False

        except asyncio.TimeoutError:
            logging.error(
                f"Lifecycle hook timed out after {hook.timeout_seconds} seconds"
            )
            all_successful = False
            if hook.on_failure == "abort":
                return False
        except Exception as e:
            logging.error(f"Error executing lifecycle hook: {e}")
            all_successful = False
            if hook.on_failure == "abort":
                return False

    return all_successful
