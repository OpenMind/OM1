"""Tests for singleton decorator."""

import threading
import time

import pytest

from providers.singleton import singleton


class TestSingletonDecorator:
    """Tests for the singleton decorator."""

    def test_single_instance_created(self):
        """Test that only one instance is created."""

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

    def test_instance_with_arguments(self):
        """Test singleton with constructor arguments."""

        @singleton
        class TestClassWithArgs:
            def __init__(self, value, name="default"):
                self.value = value
                self.name = name

        instance = TestClassWithArgs(100, name="test")

        assert instance.value == 100
        assert instance.name == "test"

        # Second call should return same instance, ignoring new args
        instance2 = TestClassWithArgs(200, name="different")
        assert instance2 is instance
        assert instance2.value == 100  # Original value preserved

        # Cleanup
        TestClassWithArgs.reset()  # type: ignore[attr-defined]

    def test_reset_instance(self):
        """Test that reset allows new instance creation."""

        @singleton
        class TestClassReset:
            def __init__(self, value):
                self.value = value

        instance1 = TestClassReset(10)
        assert instance1.value == 10

        # Reset the singleton
        TestClassReset.reset()  # type: ignore[attr-defined]

        # New instance should be created
        instance2 = TestClassReset(20)
        assert instance2.value == 20
        assert instance1 is not instance2

        # Cleanup
        TestClassReset.reset()  # type: ignore[attr-defined]

    def test_thread_safety(self):
        """Test that singleton is thread-safe."""
        creation_count = 0
        creation_lock = threading.Lock()

        @singleton
        class ThreadSafeClass:
            def __init__(self):
                nonlocal creation_count
                with creation_lock:
                    creation_count += 1
                time.sleep(0.01)

        instances = []
        errors = []

        def create_instance():
            try:
                instance = ThreadSafeClass()
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

        for instance in instances:
            assert instance is instances[0]

        assert creation_count == 1

        # Cleanup
        ThreadSafeClass.reset()  # type: ignore[attr-defined]

    def test_singleton_class_attribute(self):
        """Test that _singleton_class attribute is set."""

        @singleton
        class TestClassAttr:
            pass

        assert hasattr(TestClassAttr, "_singleton_class")

        # Cleanup
        TestClassAttr.reset()  # type: ignore[attr-defined]

    def test_multiple_singleton_classes(self):
        """Test that different classes have independent singletons."""

        @singleton
        class SingletonA:
            def __init__(self):
                self.name = "A"

        @singleton
        class SingletonB:
            def __init__(self):
                self.name = "B"

        instance_a = SingletonA()
        instance_b = SingletonB()

        assert instance_a is not instance_b
        assert instance_a.name == "A"
        assert instance_b.name == "B"

        # Cleanup
        SingletonA.reset()  # type: ignore[attr-defined]
        SingletonB.reset()  # type: ignore[attr-defined]