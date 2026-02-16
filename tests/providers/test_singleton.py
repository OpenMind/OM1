"""Tests for the singleton decorator."""

import importlib.util
import sys
import threading
from unittest.mock import MagicMock

# Mock all problematic modules before any imports
sys.modules["zenoh"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()
sys.modules["pycdr2"] = MagicMock()

# Load singleton module directly to avoid providers __init__.py chain
spec = importlib.util.spec_from_file_location("singleton", "src/providers/singleton.py")
singleton_module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(singleton_module)  # type: ignore[union-attr]
singleton = singleton_module.singleton


class TestSingleton:
    """Tests for the singleton decorator."""

    def test_singleton_returns_same_instance(self):
        """Test that singleton returns the same instance."""

        @singleton
        class TestClass:
            def __init__(self):
                self.value = 42

        instance1 = TestClass()
        instance2 = TestClass()

        assert instance1 is instance2
        assert instance1.value == 42

        # Cleanup
        TestClass.reset()  # type: ignore[attr-defined]

    def test_singleton_with_arguments(self):
        """Test singleton with constructor arguments."""

        @singleton
        class TestClassWithArgs:
            def __init__(self, value):
                self.value = value

        instance1 = TestClassWithArgs(100)
        instance2 = TestClassWithArgs(200)  # Should be ignored

        assert instance1 is instance2
        assert instance1.value == 100  # First call's value

        # Cleanup
        TestClassWithArgs.reset()  # type: ignore[attr-defined]

    def test_singleton_reset(self):
        """Test that reset creates new instance."""

        @singleton
        class TestClassReset:
            def __init__(self):
                self.value = 1

        instance1 = TestClassReset()
        instance1.value = 99

        TestClassReset.reset()  # type: ignore[attr-defined]

        instance2 = TestClassReset()

        assert instance1 is not instance2
        assert instance2.value == 1  # Fresh instance

        # Cleanup
        TestClassReset.reset()  # type: ignore[attr-defined]

    def test_singleton_thread_safety(self):
        """Test that singleton is thread-safe."""
        instances = []
        errors = []

        @singleton
        class TestClassThreadSafe:
            def __init__(self):
                self.value = threading.current_thread().name

        def create_instance():
            try:
                instance = TestClassThreadSafe()
                instances.append(instance)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_instance) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(instances) == 10
        # All instances should be the same object
        assert all(inst is instances[0] for inst in instances)

        # Cleanup
        TestClassThreadSafe.reset()  # type: ignore[attr-defined]

    def test_singleton_class_attribute(self):
        """Test that singleton exposes the original class."""

        @singleton
        class TestClassAttr:
            pass

        assert hasattr(TestClassAttr, "_singleton_class")
        assert TestClassAttr._singleton_class.__name__ == "TestClassAttr"  # type: ignore[attr-defined]

        # Cleanup
        TestClassAttr.reset()  # type: ignore[attr-defined]

    def test_singleton_with_kwargs(self):
        """Test singleton with keyword arguments."""

        @singleton
        class TestClassKwargs:
            def __init__(self, name="default", count=0):
                self.name = name
                self.count = count

        instance = TestClassKwargs(name="test", count=5)

        assert instance.name == "test"
        assert instance.count == 5

        # Cleanup
        TestClassKwargs.reset()  # type: ignore[attr-defined]

    def test_multiple_singleton_classes_independent(self):
        """Test that different singleton classes are independent."""

        @singleton
        class SingletonA:
            def __init__(self):
                self.label = "A"

        @singleton
        class SingletonB:
            def __init__(self):
                self.label = "B"

        a = SingletonA()
        b = SingletonB()

        assert a is not b
        assert a.label == "A"
        assert b.label == "B"

        # Cleanup
        SingletonA.reset()  # type: ignore[attr-defined]
        SingletonB.reset()  # type: ignore[attr-defined]
