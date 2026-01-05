import threading

from providers.singleton import singleton


@singleton
class DummyService:
    def __init__(self, value: int):
        self.value = value


def test_singleton_returns_same_instance():
    DummyService.reset()  # type: ignore[attr-defined]

    a = DummyService(1)
    b = DummyService(2)

    assert a is b
    assert a.value == 1


def test_singleton_reset_creates_new_instance():
    DummyService.reset()  # type: ignore[attr-defined]

    a = DummyService(1)
    DummyService.reset()  # type: ignore[attr-defined]
    b = DummyService(2)

    assert a is not b
    assert b.value == 2


def test_singleton_thread_safety():
    DummyService.reset()  # type: ignore[attr-defined]
    instances: list[DummyService] = []

    def create_instance():
        instances.append(DummyService(42))

    threads = [threading.Thread(target=create_instance) for _ in range(10)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(instances) == 10
    assert all(inst is instances[0] for inst in instances)