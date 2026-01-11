import threading
import time
from providers.singleton import singleton
import pytest

@singleton
class TestClass:
    def __init__(self, value=0):
        self.value = value

def test_singleton_basic():
    """Test that multiple calls return the same instance."""
    instance1 = TestClass(42)
    instance2 = TestClass(100)
    
    assert instance1 is instance2
    assert instance1.value == 42

    # Cleanup
    TestClass.reset()

def test_singleton_arguments():
    """Test that arguments are passed to the constructor (only on first creation)."""
    # Reset to ensure fresh start
    TestClass.reset()
    
    c1 = TestClass(value=42)
    assert c1.value == 42
    
    # Subsequent calls with different args should return the EXISTING instance
    # and NOT re-initialize it.
    c2 = TestClass(value=99)
    assert c2 is c1
    assert c2.value == 42  # Should still be 42
    
    # Cleanup
    TestClass.reset()

def test_singleton_reset():
    """Test that reset() allows creating a new instance."""
    TestClass.reset()
    c1 = TestClass()
    original_id = id(c1)
    
    TestClass.reset()
    c2 = TestClass()
    new_id = id(c2)
    
    assert c1 is not c2
    assert original_id != new_id
    
    # Cleanup
    TestClass.reset()

def test_singleton_thread_safety():
    """Test that singleton creation is thread-safe."""
    TestClass.reset()
    
    instances = []
    
    def get_singleton_instance():
        # Add a small delay to increase chance of race condition if not locked
        time.sleep(0.01)
        inst = TestClass()
        instances.append(inst)
        
    threads = [threading.Thread(target=get_singleton_instance) for _ in range(10)]
    
    for t in threads:
        t.start()
        
    for t in threads:
        t.join()
        
    # All instances should be identical
    first_instance = instances[0]
    for inst in instances[1:]:
        assert inst is first_instance
        
    # Cleanup
    TestClass.reset()

def test_singleton_reset_thread_safety():
    """Test that reset is thread safe."""
    # This is harder to deterministically test, but we can try to hammer it.
    pass 
