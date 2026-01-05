import threading
from typing import TypeVar, Type, Any, cast

T = TypeVar("T")


def singleton(cls: Type[T]) -> Type[T]:
    """
    Thread-safe singleton class decorator.

    Ensures that only one instance of the decorated class is created.
    The class type is preserved for static type checkers (pyright, mypy).
    """

    _lock = threading.Lock()
    _instance: T | None = None

    original_new = cls.__new__

    def __new__(inner_cls: Type[T], *args: Any, **kwargs: Any) -> T:
        nonlocal _instance
        if _instance is None:
            with _lock:
                if _instance is None:
                    _instance = cast(
                        T,
                        original_new(inner_cls)
                        if original_new is not object.__new__
                        else object.__new__(inner_cls),
                    )
        return _instance

    cls.__new__ = __new__  # type: ignore[assignment]

    def reset_instance() -> None:
        nonlocal _instance
        with _lock:
            _instance = None

    cls.reset = reset_instance  # type: ignore[attr-defined]

    return cls
