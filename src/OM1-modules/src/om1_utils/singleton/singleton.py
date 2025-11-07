import threading
from typing import Any


def singleton(cls):
    """
    A thread-safe singleton decorator that ensures only one instance of a class exists.

    This decorator implements the singleton pattern with thread safety using a lock.
    Multiple threads attempting to create an instance will be synchronized to prevent
    race conditions.

    Args:
        cls: The class to be converted into a singleton.

    Returns:
        function: A getter function that returns the singleton instance.
    """
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs) -> Any:
        """
        Returns the singleton instance of the decorated class.

        If the instance doesn't exist, creates it with the provided arguments.
        Thread-safe implementation using a lock.

        Args:
            *args: Positional arguments to pass to the class constructor.
            **kwargs: Keyword arguments to pass to the class constructor.

        Returns:
            Any: The singleton instance of the decorated class.
        """
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
