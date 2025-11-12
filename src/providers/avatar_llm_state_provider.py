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

            # Check if result contains face action
            has_face_action = False
            if result:
                actions: Optional[List[Any]] = getattr(result, "actions", None)
                if actions:
                    has_face_action = any(
                        getattr(a, "type", "").lower() == "face" for a in actions
                    )

            # Restore happy if no face action in result
            if not has_face_action and avatar_provider and avatar_provider.running:
                try:
                    avatar_provider.send_avatar_command("Happy")
                except Exception:
                    pass

            return result

        except Exception as e:
            # Restore happy on error
            if avatar_provider and avatar_provider.running:
                try:
                    avatar_provider.send_avatar_command("Happy")
                except Exception:
                    pass
            raise e

    return wrapper
