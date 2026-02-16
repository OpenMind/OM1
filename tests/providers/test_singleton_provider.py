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


def test_singleton_with_arguments():
    """Test singleton with constructor arguments - first call wins."""

    @singleton
    class ConfigManager:
        def __init__(self, config_path):
            self.config_path = config_path

    config1 = ConfigManager("/path/to/config1")
    config2 = ConfigManager("/path/to/config2")  # Should be ignored

    assert config1 is config2
    assert config1.config_path == "/path/to/config1"  # First call's value

    # Cleanup
    ConfigManager.reset()  # type: ignore


def test_singleton_with_kwargs():
    """Test singleton with keyword arguments."""

    @singleton
    class Settings:
        def __init__(self, name="default", count=0):
            self.name = name
            self.count = count

    settings = Settings(name="test", count=5)

    assert settings.name == "test"
    assert settings.count == 5

    # Cleanup
    Settings.reset()  # type: ignore


def test_singleton_class_attribute():
    """Test that singleton exposes the original class via _singleton_class."""

    @singleton
    class ServiceManager:
        pass

    assert hasattr(ServiceManager, "_singleton_class")
    assert ServiceManager._singleton_class.__name__ == "ServiceManager"  # type: ignore

    # Cleanup
    ServiceManager.reset()  # type: ignore


def test_multiple_singleton_classes_independent():
    """Test that different singleton classes remain independent."""

    @singleton
    class CacheA:
        def __init__(self):
            self.label = "A"

    @singleton
    class CacheB:
        def __init__(self):
            self.label = "B"

    a = CacheA()
    b = CacheB()

    assert a is not b
    assert a.label == "A"
    assert b.label == "B"

    # Cleanup
    CacheA.reset()  # type: ignore
    CacheB.reset()  # type: ignore
