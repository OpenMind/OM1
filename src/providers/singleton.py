import threading
from functools import wraps
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    cast,
)

T = TypeVar("T")


class SingletonCallable(Protocol[T]):
    """
    Protocol for a singleton callable with a reset hook.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> T: ...
    def reset(self) -> None: ...


def singleton(cls: Type[T]) -> SingletonCallable[T]:
    """
    Thread-safe singleton decorator.

    Ensures that only one instance of the decorated class is created.
    Subsequent calls return the same instance.

    Features:
    - Thread-safe initialization
    - Type-safe (generic)
    - Preserves class metadata
    - Detects inconsistent constructor arguments
    - Supports reset for testing
    """
    instance: Optional[T] = None
    init_args: Optional[Tuple[Tuple[Any, ...], dict[str, Any]]] = None
    lock = threading.Lock()

    @wraps(cls)
    def get_instance(*args: Any, **kwargs: Any) -> T:
        nonlocal instance, init_args

        # Fast path
        if instance is not None:
            if init_args != (args, kwargs):
                raise RuntimeError(
                    f"Singleton '{cls.__name__}' already initialized with "
                    f"different arguments"
                )
            return instance

        # Slow path
        with lock:
            if instance is None:
                instance = cls(*args, **kwargs)
                init_args = (args, kwargs)
            elif init_args != (args, kwargs):
                raise RuntimeError(
                    f"Singleton '{cls.__name__}' already initialized with "
                    f"different arguments"
                )

        return instance

    def reset_instance() -> None:
        """
        Reset the singleton instance.

        Intended for testing purposes only.
        """
        nonlocal instance, init_args
        with lock:
            instance = None
            init_args = None

    get_instance.reset = reset_instance  # type: ignore[attr-defined]

    return cast(SingletonCallable[T], get_instance)
