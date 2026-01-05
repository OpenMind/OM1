import threading
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class SingletonCallable(Generic[T]):
    """
    Callable wrapper returned by the @singleton decorator.
    """

    _singleton_class: type[T]

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Return the singleton instance."""
        ...

    def reset(self) -> None:
        """Reset the singleton instance."""
        ...


def singleton(cls: type[T]) -> Callable[..., T]:
    """
    Thread-safe singleton decorator.

    Ensures only one instance of the decorated class exists.
    """
    lock = threading.Lock()
    instance: T | None = None

    def get_instance(*args: Any, **kwargs: Any) -> T:
        """
        Return the singleton instance, creating it if necessary.
        """
        nonlocal instance
        with lock:
            if instance is None:
                instance = cls(*args, **kwargs)
            return instance

    def reset() -> None:
        """
        Reset the stored singleton instance.
        """
        nonlocal instance
        with lock:
            instance = None

    # typing + runtime metadata
    get_instance.reset = reset  # type: ignore[attr-defined]
    get_instance._singleton_class = cls  # type: ignore[attr-defined]

    return get_instance
