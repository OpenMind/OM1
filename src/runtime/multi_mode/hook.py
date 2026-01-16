import asyncio
import importlib
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider


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


class LifecycleHookHandler:
    """
    Base class for lifecycle hook handlers.
    """

    def __init__(self, config: Dict[str, Any]):
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
        message = self.config.get("message", "")
        if message:
            try:
                formatted_message = message.format(**context)
                logging.info(f"Lifecycle hook message: {formatted_message}")

                try:
                    ElevenLabsTTSProvider().add_pending_message(formatted_message)
                except Exception as e:
                    logging.error(f"Error adding TTS message: {e}")
                    return False

                return True
            except Exception as e:
                logging.error(f"Error formatting lifecycle message: {e}")
                return False
        return True


class CommandHookHandler(LifecycleHookHandler):
    """
    Handler that executes a shell command.
    """

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
        command = self.config.get("command", "")
        if not command:
            logging.warning("No command specified for command hook")
            return False

        try:
            formatted_command = command.format(**context)

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
        module_name = self.config.get("module_name")
        function_name = self.config.get("function")

        if not function_name:
            logging.error("No function specified for function hook")
            return False

        if not module_name:
            logging.error("No module_name specified for function hook")
            return False

        try:
            func = self._find_function_in_module(module_name, function_name)
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

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
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
            action_type = self.config.get("action_type")
            if not action_type:
                logging.error("No action_type specified for action hook")
                return False

            action_config = self.config.get("action_config", {})
            try:
                from actions import load_action

                self.action = load_action(
                    {"type": action_type, "config": action_config}
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

    if handler_type == "message":
        return MessageHookHandler(hook.handler_config)
    elif handler_type == "command":
        return CommandHookHandler(hook.handler_config)
    elif handler_type == "function":
        return FunctionHookHandler(hook.handler_config)
    elif handler_type == "action":
        return ActionHookHandler(hook.handler_config)
    else:
        logging.error(f"Unknown hook handler type: {handler_type}")
        return None


def parse_lifecycle_hooks(raw_hooks: List[Dict]) -> List[LifecycleHook]:
    """
    Parse raw lifecycle hooks configuration into LifecycleHook objects.

    Parameters
    ----------
    raw_hooks : List[Dict]
        Raw hook configuration data

    Returns
    -------
    List[LifecycleHook]
        Parsed lifecycle hook objects
    """
    hooks = []
    for hook_data in raw_hooks:
        try:
            hook = LifecycleHook(
                hook_type=LifecycleHookType(hook_data["hook_type"]),
                handler_type=hook_data["handler_type"],
                handler_config=hook_data.get("handler_config", {}),
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
