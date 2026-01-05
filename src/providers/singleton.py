import threading
from typing import Any, Type, TypeVar, cast

T = TypeVar("T")


def singleton(cls: Type[T]) -> Type[T]:
    cls._instance: T | None = None  # type: ignore[attr-defined]
    _lock = threading.Lock()

    original_new = cls.__new__

    def __new__(inner_cls, *args: Any, **kwargs: Any) -> T:
        if inner_cls._instance is None:
            with _lock:
                if inner_cls._instance is None:
                    instance = original_new(inner_cls)
                    inner_cls._instance = cast(T, instance)
        return inner_cls._instance  # type: ignore[return-value]

    def _reset() -> None:
        with _lock:
            cls._instance = None  # type: ignore[attr-defined]

    cls.__new__ = __new__  # type: ignore[method-assign]
    setattr(cls, "reset", staticmethod(_reset))

    return cls
