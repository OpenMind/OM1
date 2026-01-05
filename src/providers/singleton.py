import threading
from functools import wraps
from typing import Any, Optional, Protocol, Tuple, Type, TypeVar, cast

T = TypeVar("T")


class SingletonCallable(Protocol[T]):
    def __call__(self, *args: Any, **kwargs: Any) -> T: ...
    def reset(self) -> None: ...


def singleton(cls: Type[T]) -> SingletonCallable[T]:
    """
    Thread-safe singleton decorator.

    Ensures only one instance of the decorated class exists.
    Detects inconsistent initialization arguments and supports reset for tests.
    """
    instance: Optional[T] = None
    init_args: Optional[Tuple[Tuple[Any, ...], dict[str, Any]]] = None
    lock = threading.Lock()

    @wraps(cls)
    def get_instance(*args: Any, **kwargs: Any) -> T:
        nonlocal instance, init_args

        if instance is not None:
            if init_args != (args, kwargs):
                raise RuntimeError(
                    f"Singleton '{cls.__name__}' already initialized "
                    f"with different arguments"
                )
            return instance

        with lock:
            if instance is None:
                instance = cls(*args, **kwargs)
                init_args = (args, kwargs)
            elif init_args != (args, kwargs):
                raise RuntimeError(
                    f"Singleton '{cls.__name__}' already initialized "
                    f"with different arguments"
                )

        return instance

    def reset_instance() -> None:
        nonlocal instance, init_args
        with lock:
            instance = None
            init_args = None

    get_instance.reset = reset_instance  # type: ignore[attr-defined]

    return cast(SingletonCallable[T], get_instance)
