import functools
import logging
import threading
from typing import Any, Awaitable, Callable, List, Optional, TypeVar

from .avatar_provider import AvatarProvider
from .io_provider import IOProvider

T = TypeVar("T")


class AvatarLLMState:
    """
    Singleton class to manage avatar thinking state during LLM processing.
    """

    _instance = None
    _lock = None

    def __new__(cls):
        """
        Implement singleton pattern for AvatarLLMState.
        """
        if cls._instance is None:
            if cls._lock is None:
                cls._lock = threading.Lock()

            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialize the AvatarLLMState singleton instance.
        """
        if not getattr(self, "_initialized", False):
            self.avatar_provider: Optional[AvatarProvider] = None
            self.io_provider: Optional[IOProvider] = None
            try:
                self.avatar_provider = AvatarProvider()
            except Exception:
                logging.error("Failed to initialize AvatarProvider in AvatarLLMState")
                self.avatar_provider = None
            try:
                self.io_provider = IOProvider()
            except Exception:
                logging.error("Failed to initialize IOProvider in AvatarLLMState")
                self.io_provider = None
            self._initialized = True

    def _has_voice_input(self) -> bool:
        """
        Check if current input contains voice input.

        Returns
        -------
        bool
            True if voice input is present, False otherwise
        """
        if not self.io_provider:
            return False

        try:
            return (
                self.io_provider.llm_prompt is not None
                and "INPUT: Voice" in self.io_provider.llm_prompt
            )
        except Exception:
            return False

    def _start_thinking(self) -> None:
        """
        Internal method to trigger the thinking animation on the avatar.

        Sets the avatar to "Think" state to indicate LLM processing.
        """
        if not self._has_voice_input():
            return

        if self.avatar_provider and self.avatar_provider.running:
            try:
                self.avatar_provider.send_avatar_command("Think")
            except Exception:
                logging.error(
                    "Failed to send 'Think' command to avatar in AvatarLLMState",
                    exc_info=True,
                )

    def _restore_happy(self) -> None:
        """
        Restore the avatar to happy state.

        Sets the avatar to "Happy" state after processing completion.
        """
        if self.avatar_provider and self.avatar_provider.running:
            try:
                self.avatar_provider.send_avatar_command("Happy")
            except Exception:
                pass

    def _has_face_action_in_result(self, result: Any) -> bool:
        """
        Check if the result contains a face action.

        Parameters
        ----------
        result : Any
            The result object to check.

        Returns
        -------
        bool
            True if result contains a face action, False otherwise
        """
        if not result:
            return False

        actions: Optional[List[Any]] = getattr(result, "actions", None)
        if not actions:
            return False

        return any(getattr(a, "type", "").lower() == "face" for a in actions)

    @classmethod
    def trigger_thinking(
        cls, func: Optional[Callable[..., Awaitable[T]]] = None
    ) -> Any:
        """
        Decorator to manage avatar state during LLM processing.

        Parameters
        ----------
        func : Optional[Callable[..., Awaitable[T]]]
            The async function to wrap.

        Returns
        -------
        Callable or wrapped function
            Wrapped function that manages avatar state
        """

        def decorator(f: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @functools.wraps(f)
            async def wrapper(*args: Any, **kwargs: Any) -> T:

                if getattr(args[0], "_skip_state_management", False):
                    return await f(*args, **kwargs)

                instance = cls()
                instance._start_thinking()

                try:
                    result = await f(*args, **kwargs)

                    if not instance._has_face_action_in_result(result):
                        instance._restore_happy()

                    return result

                except Exception as e:
                    instance._restore_happy()
                    raise e

            return wrapper

        if func is not None:
            return decorator(func)

        return decorator

    def stop(self) -> None:
        """
        Stop the AvatarLLMState singleton instance.
        """
        if self.avatar_provider and self.avatar_provider.running:
            try:
                self.avatar_provider.stop()
            except Exception:
                logging.error("Failed to stop AvatarProvider in AvatarLLMState")
