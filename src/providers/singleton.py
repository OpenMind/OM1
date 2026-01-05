import threading
from typing import Any, Protocol, TypeVar, cast

T = TypeVar("T")


class SingletonCallable(Protocol[T]):
    """
    Callable singleton wrapper that still behaves like a class for typing.
    """

    __name__: str
    __qualname__: str

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Return the singleton instance."""
        ...

    def reset(self) -> None:
        """Reset the singleton instance."""
        ...


def singleton(cls: type[T]) -> SingletonCallable[T]:
    """
    Thread-safe singleton decorator that preserves class typing.
    """
    instance: T | None = None
    lock = threading.Lock()

    def get_instance(*args: Any, **kwargs: Any) -> T:
        nonlocal instance
        with lock:
            if instance is None:
                instance = cls(*args, **kwargs)
            return instance

    def reset() -> None:
        nonlocal instance
        with lock:
            instance = None

    get_instance.reset = reset  # type: ignore[attr-defined]
    get_instance.__name__ = cls.__name__
    get_instance.__qualname__ = cls.__qualname__

    return cast(SingletonCallable[T], get_instance)
