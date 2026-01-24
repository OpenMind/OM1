import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def singleton(cls: type[T]) -> Callable[..., T]:
    """
    A thread-safe singleton decorator that ensures only one instance of a class exists.

    This decorator implements a singleton pattern with thread safety using a lock.
    Multiple threads attempting to create an instance will be synchronized to prevent
    race conditions.

    All instance checks and creation are performed within the lock to ensure
    thread safety without relying on double-checked locking patterns.

    Args:
        cls: The class to be converted into a singleton.

    Returns
    -------
        function: A getter function that returns the singleton instance.

    Example
    -------
        @singleton
        class MyService:
            def __init__(self, config: str):
                self.config = config

        # First call creates the instance
        service1 = MyService("config_value")
        # Subsequent calls return the same instance
        service2 = MyService("different_value")
        assert service1 is service2
    """
    lock = threading.Lock()
    instance: T | None = None

    def get_instance(*args: Any, **kwargs: Any) -> T:
        """
        Returns the singleton instance of the decorated class.

        If the instance doesn't exist, creates it with the provided arguments.
        Thread-safe implementation using a lock.

        Args:
            *args: Positional arguments to pass to the class constructor.
            **kwargs: Keyword arguments to pass to the class constructor.

        Returns
        -------
            T: The singleton instance of the decorated class.
        """
        nonlocal instance
        with lock:
            if instance is None:
                instance = cls(*args, **kwargs)
            return instance

    def reset_instance() -> None:
        """
        Resets the singleton instance of the decorated class.

        This method sets the singleton instance to None, allowing a new instance
        to be created on the next call to get_instance.
        """
        nonlocal instance
        with lock:
            instance = None

    get_instance._singleton_class = cls  # type: ignore[attr-defined]
    get_instance.reset = reset_instance  # type: ignore[attr-defined]

    return get_instance
