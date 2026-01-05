
import threading
from typing import Any, Type, TypeVar, cast

T = TypeVar("T")


def singleton(cls: Type[T]) -> Type[T]:
    setattr(cls, "_instance", None)
    setattr(cls, "_initialized", False)
    _lock = threading.Lock()

    original_new = cls.__new__
    original_init = cls.__init__

    def __new__(inner_cls, *args: Any, **kwargs: Any) -> T:
        if getattr(inner_cls, "_instance") is None:
            with _lock:
                if getattr(inner_cls, "_instance") is None:
                    instance = original_new(inner_cls)
                    setattr(inner_cls, "_instance", cast(T, instance))
        return cast(T, getattr(inner_cls, "_instance"))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if getattr(cls, "_initialized"):
            return
        with _lock:
            if not getattr(cls, "_initialized"):
                original_init(self, *args, **kwargs)
                setattr(cls, "_initialized", True)

    def _reset() -> None:
        with _lock:
            setattr(cls, "_instance", None)
            setattr(cls, "_initialized", False)

    cls.__new__ = __new__  # type: ignore[method-assign]
    cls.__init__ = __init__  # type: ignore[method-assign]
    setattr(cls, "reset", staticmethod(_reset))

    return cls