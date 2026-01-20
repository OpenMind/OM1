import threading
from typing import Any


def singleton(cls):
    """
    A thread-safe singleton decorator that ensures only one instance of a class exists.

    This decorator implements a singleton pattern with thread safety using a lock.
    Multiple threads attempting to create an instance will be synchronized to prevent
    race conditions.

    Args:
        cls: The class to be converted into a singleton.

    Returns
    -------
        function: A getter function that returns the singleton instance.
    """
    if not hasattr(cls, "_singleton_instance"):
        cls._singleton_instance = None
    lock = threading.Lock()

    def get_instance(*args, **kwargs) -> Any:
        """
        Returns the singleton instance of the decorated class.

        If the instance doesn't exist, creates it with the provided arguments.
        Thread-safe implementation using a lock.

        Args:
            *args: Positional arguments to pass to the class constructor.
            **kwargs: Keyword arguments to pass to the class constructor.

        Returns
        -------
            Any: The singleton instance of the decorated class.
        """
        with lock:
            if cls._singleton_instance is None:
                cls._singleton_instance = cls(*args, **kwargs)
            return cls._singleton_instance

    def reset_instance():
        """
        Resets the singleton instance of the decorated class.

        This method sets the singleton instance to None, allowing a new instance
        to be created on the next call to get_instance.
        """
        with lock:
            cls._singleton_instance = None

    get_instance._singleton_class = cls  # type: ignore
    get_instance.reset = reset_instance  # type: ignore

    return get_instance
