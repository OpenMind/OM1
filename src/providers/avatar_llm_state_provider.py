import functools
from typing import Any, Awaitable, Callable, List, Optional, TypeVar

from .avatar_provider import AvatarProvider

T = TypeVar("T")


def AvatarLLMStateProvider(
    func: Callable[..., Awaitable[T]],
) -> Callable[..., Awaitable[T]]:
    """
    Decorator to manage avatar thinking state during LLM processing.

    Sets avatar to "Think" state when LLM starts processing,
    Restores to "Happy" state after completion if no face action was generated.

    """

    @functools.wraps(func)
    async def wrapper(self, *args: Any, **kwargs: Any) -> T:
        # Set thinking state before LLM processing
        avatar_provider = None
        try:
            avatar_provider = AvatarProvider()
            if avatar_provider.running:
                avatar_provider.send_avatar_command("Think")
        except Exception:
            pass

        try:
            result = await func(self, *args, **kwargs)

            # Restore happy if no face action in result
            if result and avatar_provider:
                actions: Optional[List[Any]] = getattr(result, "actions", None)
                if actions:
                    has_face = any(
                        getattr(a, "type", "").lower() == "face" for a in actions
                    )
                    if not has_face:
                        try:
                            avatar_provider.send_avatar_command("Happy")
                        except Exception:
                            pass

            return result

        except Exception as e:
            if avatar_provider:
                try:
                    avatar_provider.send_avatar_command("Happy")
                except Exception:
                    pass
            raise e

    return wrapper
