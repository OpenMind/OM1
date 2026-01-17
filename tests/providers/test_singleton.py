"""Tests for singleton."""

import sys
import threading
import time
from unittest.mock import MagicMock

import pytest

# Mock ALL external dependencies BEFORE any provider imports
# This must happen at module load time
sys.modules["zenoh"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["rclpy"] = MagicMock()
sys.modules["rclpy.node"] = MagicMock()
sys.modules["rclpy.qos"] = MagicMock()
sys.modules["sensor_msgs"] = MagicMock()
sys.modules["sensor_msgs.msg"] = MagicMock()
sys.modules["geometry_msgs"] = MagicMock()
sys.modules["geometry_msgs.msg"] = MagicMock()
sys.modules["nav_msgs"] = MagicMock()
sys.modules["nav_msgs.msg"] = MagicMock()
sys.modules["std_msgs"] = MagicMock()
sys.modules["std_msgs.msg"] = MagicMock()
sys.modules["elevenlabs"] = MagicMock()
sys.modules["riva"] = MagicMock()
sys.modules["riva.client"] = MagicMock()
sys.modules["pyaudio"] = MagicMock()
sys.modules["sounddevice"] = MagicMock()
sys.modules["websocket"] = MagicMock()
sys.modules["websockets"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()


class TestSingleton:
    """Tests for singleton decorator."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        # Clear cached provider modules to reset singletons
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        yield
        # Cleanup after test
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

    def test_singleton_decorator_creates_instance(self):
        """Test singleton decorator creates an instance correctly."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self, value):
                self.value = value

        instance = TestClass("test")
        assert instance is not None
        assert instance.value == "test"

    def test_singleton_decorator_returns_same_instance(self):
        """Test singleton decorator returns the same instance on multiple calls."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self, value):
                self.value = value

        instance1 = TestClass("test1")
        instance2 = TestClass("test2")

        assert instance1 is instance2
        assert instance1.value == "test1"  # First instance's value is preserved

    def test_singleton_decorator_with_no_args(self):
        """Test singleton decorator works with classes that take no arguments."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self):
                self.value = "default"

        instance1 = TestClass()
        instance2 = TestClass()

        assert instance1 is instance2
        assert instance1.value == "default"

    def test_singleton_decorator_reset_functionality(self):
        """Test singleton decorator's reset functionality."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self, value):
                self.value = value

        instance1 = TestClass("test1")
        TestClass.reset()
        instance2 = TestClass("test2")

        assert instance1 is not instance2
        assert instance1.value == "test1"
        assert instance2.value == "test2"

    def test_singleton_decorator_thread_safety(self):
        """Test singleton decorator is thread-safe."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self, value):
                self.value = value
                # Simulate some work during initialization
                time.sleep(0.01)

        TestClass.reset()

        instances = []

        def create_instance(value):
            instances.append(TestClass(value))

        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_instance, args=(f"test{i}",))
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All instances should be the same
        first_instance = instances[0]
        for instance in instances[1:]:
            assert instance is first_instance

    def test_singleton_decorator_preserves_class_attributes(self):
        """Test singleton decorator preserves class attributes and methods."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            class_var = "class_value"

            def __init__(self, value):
                self.value = value

            def get_value(self):
                return self.value

            @classmethod
            def class_method(cls):
                return cls.class_var

        instance = TestClass("test")

        assert hasattr(TestClass, "_singleton_class")
        assert instance.get_value() == "test"
        assert TestClass._singleton_class.class_var == "class_value"
        assert TestClass._singleton_class.class_method() == "class_value"

    def test_singleton_decorator_reset_thread_safety(self):
        """Test singleton decorator reset is thread-safe."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self, value):
                self.value = value

        TestClass.reset()

        instances = []

        def create_and_reset(value):
            instance = TestClass(value)
            instances.append(instance)
            TestClass.reset()

        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_and_reset, args=(f"test{i}",))
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should not raise any exceptions
        assert len(instances) == 5

    def test_singleton_decorator_multiple_classes(self):
        """Test singleton decorator works with multiple different classes."""
        from providers.singleton import singleton

        @singleton
        class TestClass1:
            def __init__(self, value):
                self.value = value

        @singleton
        class TestClass2:
            def __init__(self, value):
                self.value = value

        instance1a = TestClass1("test1a")
        instance1b = TestClass1("test1b")
        instance2a = TestClass2("test2a")
        instance2b = TestClass2("test2b")

        assert instance1a is instance1b
        assert instance2a is instance2b
        assert instance1a is not instance2a
        assert instance1a.value == "test1a"
        assert instance2a.value == "test2a"

    def test_singleton_decorator_with_kwargs(self):
        """Test singleton decorator works with keyword arguments."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self, value, name="default"):
                self.value = value
                self.name = name

        instance1 = TestClass("test1", name="first")
        instance2 = TestClass("test2", name="second")

        assert instance1 is instance2
        assert instance1.value == "test1"
        assert instance1.name == "first"

    def test_singleton_decorator_reset_allows_new_instance(self):
        """Test reset allows creation of new instance with different parameters."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self, value):
                self.value = value

        instance1 = TestClass("original")
        original_id = id(instance1)

        TestClass.reset()

        instance2 = TestClass("new")
        new_id = id(instance2)

        assert original_id != new_id
        assert instance2.value == "new"
