import threading
import time

from providers.singleton import singleton


def test_singleton_basic_functionality():
    """Test that two calls to the decorated class return the exact same instance."""

    @singleton
    class DatabaseConnection:
        def __init__(self):
            self.id = id(self)

    db1 = DatabaseConnection()
    db2 = DatabaseConnection()
    assert db1 is db2
    assert db1.id == db2.id


def test_singleton_reset():
    """Test that calling .reset() allows a new instance to be created."""

    @singleton
    class Logger:
        pass

    log1 = Logger()
    Logger.reset()  # type: ignore
    log2 = Logger()

    assert log1 is not log2


def test_singleton_thread_safety():
    """Test that the singleton is thread-safe even with concurrent access."""

    @singleton
    class SharedResource:
        def __init__(self):
            time.sleep(0.01)

    instances = []

    def create_instance():
        instances.append(SharedResource())

    threads = [threading.Thread(target=create_instance) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    first_instance = instances[0]
    for inst in instances:
        assert inst is first_instance
