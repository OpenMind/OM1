import importlib.util
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

# Load singleton module directly without triggering providers/__init__.py
spec = importlib.util.spec_from_file_location(
    "singleton", "src/providers/singleton.py"
)
singleton_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(singleton_module)
singleton = singleton_module.singleton


class TestSingletonDecorator:
    """Test suite for the singleton decorator."""

    def setup_method(self):
        """Reset singleton state before each test."""
        # Clean up any existing singleton instances
        pass

    def test_single_instance_creation(self):
        """Test that only one instance is created."""

        @singleton
        class TestClass:
            def __init__(self, value: int = 0):
                self.value = value

        TestClass.reset()

        instance1 = TestClass(42)
        instance2 = TestClass(100)

        assert instance1 is instance2
        assert instance1.value == 42
        assert instance2.value == 42

        TestClass.reset()

    def test_reset_creates_new_instance(self):
        """Test that reset allows creating a new instance."""

        @singleton
        class TestClass:
            def __init__(self, value: int = 0):
                self.value = value

        TestClass.reset()

        instance1 = TestClass(10)
        assert instance1.value == 10

        TestClass.reset()

        instance2 = TestClass(20)
        assert instance2.value == 20
        assert instance1 is not instance2

        TestClass.reset()

    def test_thread_safety(self):
        """Test that singleton is thread-safe."""

        @singleton
        class Counter:
            def __init__(self):
                self.count = 0
                self.creation_count = 0
                self.creation_count += 1

        Counter.reset()

        instances = []
        errors = []

        def get_instance():
            try:
                instance = Counter()
                instances.append(instance)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(100):
            t = threading.Thread(target=get_instance)
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(instances) == 100
        assert all(inst is instances[0] for inst in instances)
        assert instances[0].creation_count == 1

        Counter.reset()

    def test_singleton_with_args_and_kwargs(self):
        """Test singleton with various argument types."""

        @singleton
        class ConfigClass:
            def __init__(self, name: str, value: int = 0, enabled: bool = True):
                self.name = name
                self.value = value
                self.enabled = enabled

        ConfigClass.reset()

        instance = ConfigClass("test", value=42, enabled=False)

        assert instance.name == "test"
        assert instance.value == 42
        assert instance.enabled is False

        # Second call should return same instance regardless of args
        instance2 = ConfigClass("different", value=100, enabled=True)
        assert instance2 is instance
        assert instance2.name == "test"

        ConfigClass.reset()

    def test_singleton_class_attribute(self):
        """Test thatsingleton_class attribute is set correctly."""

        @singleton
        class MyClass:
            pass

        assert hasattr(MyClass, "_singleton_class")
        assert MyClass._singleton_class.__name__ == "MyClass"

        MyClass.reset()

    def test_multiple_singleton_classes_independent(self):
        """Test that different singleton classes are independent."""

        @singleton
        class ClassA:
            def __init__(self):
                self.name = "A"

        @singleton
        class ClassB:
            def __init__(self):
                self.name = "B"

        ClassA.reset()
        ClassB.reset()

        instance_a = ClassA()
        instance_b = ClassB()

        assert instance_a is not instance_b
        assert instance_a.name == "A"
        assert instance_b.name == "B"

        ClassA.reset()
        ClassB.reset()

    def test_concurrent_reset_and_get(self):
        """Test concurrent reset and get operations."""

        @singleton
        class TestClass:
            def __init__(self):
                self.created_at = time.time()

        TestClass.reset()

        results = {"instances": [], "errors": []}

        def worker(should_reset: bool):
            try:
                if should_reset:
                    TestClass.reset()
                    time.sleep(0.001)
                instance = TestClass()
                results["instances"].append(instance)
            except Exception as e:
                results["errors"].append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(50):
                futures.append(executor.submit(worker, i % 10 == 0))

            for f in futures:
                f.result()

        assert len(results["errors"]) == 0

        TestClass.reset()
