import threading
from typing import Any, Callable, TypeVar, cast

T = TypeVar("T")


def singleton(cls: type[T]) -> Callable[..., T]:
    lock = threading.Lock()
    instance: dict[str, T] = {}

    def get_instance(*args: Any, **kwargs: Any) -> T:
        with lock:
            if "value" not in instance:
                instance["value"] = cls(*args, **kwargs)
            return instance["value"]

    def reset() -> None:
        with lock:
            instance.clear()

    setattr(get_instance, "reset", reset)
    setattr(get_instance, "_singleton_class", cls)

    return cast(Callable[..., T], get_instance)
    __all__ = ["singleton"]
