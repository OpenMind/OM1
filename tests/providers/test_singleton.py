"""
Unit tests for the singleton decorator in providers/singleton.py.

Tests cover:
- Basic singleton behavior (same instance returned)
- Thread safety
- Reset functionality
- Multiple decorated classes remain independent
- Constructor arguments handling
"""

import threading

from providers.singleton import singleton


class TestSingletonBasicBehavior:
    """Tests for basic singleton pattern functionality."""

    def test_returns_same_instance(self):
        """Test that the singleton decorator returns the same instance on multiple calls."""
        @singleton
        class TestClass:
            def __init__(self):
                self.value = 42

        instance1 = TestClass()
        instance2 = TestClass()

        assert instance1 is instance2
        assert instance1.value == 42

        # Cleanup
        TestClass.reset()

    def test_instance_preserves_state(self):
        """Test that modifications to the singleton persist across calls."""
        @singleton
        class StatefulClass:
            def __init__(self):
                self.counter = 0

            def increment(self):
                self.counter += 1

        instance1 = StatefulClass()
        instance1.increment()
        instance1.increment()

        instance2 = StatefulClass()
        assert instance2.counter == 2  # Same instance, counter should be 2

        # Cleanup
        StatefulClass.reset()


class TestSingletonReset:
    """Tests for the reset functionality."""

    def test_reset_creates_new_instance(self):
        """Test that reset() allows a new instance to be created."""
        @singleton
        class ResettableClass:
            def __init__(self):
                self.id = id(self)

        instance1 = ResettableClass()
        original_id = instance1.id

        ResettableClass.reset()

        instance2 = ResettableClass()
        assert instance2.id != original_id
        assert instance1 is not instance2

        # Cleanup
        ResettableClass.reset()

    def test_reset_clears_state(self):
        """Test that reset clears the previous instance state."""
        @singleton
        class CounterClass:
            def __init__(self):
                self.count = 0

        instance = CounterClass()
        instance.count = 100

        CounterClass.reset()

        new_instance = CounterClass()
        assert new_instance.count == 0  # Fresh instance

        # Cleanup
        CounterClass.reset()


class TestSingletonConstructorArgs:
    """Tests for constructor argument handling."""

    def test_first_call_args_are_used(self):
        """Test that only the first call's arguments are used for initialization."""
        @singleton
        class ConfigClass:
            def __init__(self, value=0):
                self.value = value

        instance1 = ConfigClass(value=42)
        instance2 = ConfigClass(value=100)  # This should be ignored

        assert instance1.value == 42
        assert instance2.value == 42  # Same instance, original value

        # Cleanup
        ConfigClass.reset()

    def test_kwargs_work_correctly(self):
        """Test that keyword arguments work for singleton initialization."""
        @singleton
        class KwargsClass:
            def __init__(self, name="default", count=0):
                self.name = name
                self.count = count

        instance = KwargsClass(name="test", count=5)

        assert instance.name == "test"
        assert instance.count == 5

        # Cleanup
        KwargsClass.reset()


class TestSingletonThreadSafety:
    """Tests for thread-safe behavior."""

    def test_thread_safe_initialization(self):
        """Test that singleton is thread-safe during initialization."""
        init_count = {"count": 0}
        init_lock = threading.Lock()

        @singleton
        class ThreadSafeClass:
            def __init__(self):
                with init_lock:
                    init_count["count"] += 1
                self.value = "initialized"

        instances = []
        errors = []

        def create_instance():
            try:
                inst = ThreadSafeClass()
                instances.append(inst)
            except Exception as e:
                errors.append(e)

        # Create multiple threads trying to instantiate simultaneously
        threads = [threading.Thread(target=create_instance) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert init_count["count"] == 1  # Only one initialization
        assert all(inst is instances[0] for inst in instances)  # All same instance

        # Cleanup
        ThreadSafeClass.reset()


class TestSingletonMultipleClasses:
    """Tests for multiple singleton classes coexisting."""

    def test_independent_singletons(self):
        """Test that different singleton classes maintain independent instances."""
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

        assert instance_a.name == "A"
        assert instance_b.name == "B"
        assert instance_a is not instance_b

        # Cleanup
        SingletonA.reset()
        SingletonB.reset()


class TestSingletonClassAttribute:
    """Tests for the _singleton_class attribute."""

    def test_singleton_class_attribute_exists(self):
        """Test that the decorated function has _singleton_class attribute."""
        @singleton
        class MarkedClass:
            pass

        assert hasattr(MarkedClass, "_singleton_class")

        # Cleanup
        MarkedClass.reset()

    def test_reset_method_exists(self):
        """Test that the decorated function has reset method."""
        @singleton
        class ResetableClass:
            pass

        assert hasattr(ResetableClass, "reset")
        assert callable(ResetableClass.reset)

        # Cleanup
        ResetableClass.reset()
