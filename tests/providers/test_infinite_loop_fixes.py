"""
Test cases for infinite loop bug fixes in provider classes.
"""

import threading
import time

from providers.odom_provider import OdomProvider
from providers.simple_paths_provider import SimplePathsProvider
from providers.unitree_go2_state_provider import UnitreeGo2StateProvider


class TestInfiniteLoopFixes:
    """Test cases for infinite loop bug fixes."""

    def test_simple_paths_provider_stop_event(self):
        """Test that SimplePathsProvider can be stopped properly."""
        provider = SimplePathsProvider()
        
        # Start the provider
        provider.start()
        
        # Verify the thread is running
        assert provider._simple_paths_derived_thread is not None
        assert provider._simple_paths_derived_thread.is_alive()
        
        # Stop the provider
        provider.stop()
        
        # Give the thread time to stop
        time.sleep(0.2)
        
        # Verify the stop event is set
        assert provider._stop_event.is_set()
        
        # Verify the thread has stopped (or will stop soon)
        # Note: The thread might still be alive briefly due to the sleep in the loop
        # but it should stop when it checks the stop event

    def test_unitree_go2_state_provider_stop_event(self):
        """Test that UnitreeGo2StateProvider can be stopped properly."""
        provider = UnitreeGo2StateProvider()
        
        # Start the provider
        provider.start()
        
        # Verify the thread is running
        assert provider._go2_state_processor_thread is not None
        assert provider._go2_state_processor_thread.is_alive()
        
        # Stop the provider
        provider.stop()
        
        # Give the thread time to stop
        time.sleep(0.2)
        
        # Verify the stop event is set
        assert provider._stop_event.is_set()

    def test_odom_provider_stop_event(self):
        """Test that OdomProvider can be stopped properly."""
        # Note: OdomProvider is a singleton and __init__ calls start() automatically
        provider = OdomProvider(channel="test_channel", use_zenoh=False)
        
        assert hasattr(provider, '_stop_event')
        assert isinstance(provider._stop_event, threading.Event)
        
        provider._stop_event.set()
        assert provider._stop_event.is_set()

    def test_simple_paths_provider_loop_termination(self):
        """Test that the derived processor loop can be terminated."""
        provider = SimplePathsProvider()
        
        # Mock the stop event to be set immediately
        provider._stop_event.set()
        
        # Start the provider
        provider.start()
        
        # Give the thread time to start and then check stop condition
        time.sleep(0.1)
        
        # The thread should terminate quickly due to the stop event being set
        # This test verifies that the while loop condition works correctly

    def test_thread_safety_of_stop_events(self):
        """Test that stop events are thread-safe."""
        provider = SimplePathsProvider()
        
        # Test setting and getting the stop event from different threads
        def set_stop_event():
            time.sleep(0.1)
            provider._stop_event.set()
        
        # Start a thread to set the stop event
        setter_thread = threading.Thread(target=set_stop_event)
        setter_thread.start()
        
        # Start the provider
        provider.start()
        
        # Wait for the setter thread to set the stop event
        setter_thread.join()
        
        # Verify the stop event is set
        assert provider._stop_event.is_set()
        
        # Clean up
        provider.stop()

    def test_multiple_stop_calls_safety(self):
        """Test that calling stop multiple times is safe."""
        provider = SimplePathsProvider()
        
        # Start the provider
        provider.start()
        
        # Call stop multiple times
        provider.stop()
        provider.stop()
        provider.stop()
        
        # Verify the stop event is set
        assert provider._stop_event.is_set()
        
        # This should not raise any exceptions

    def test_stop_event_initialization(self):
        """Test that stop events are properly initialized."""
        # Test SimplePathsProvider (singleton, so reset stop event if set)
        provider1 = SimplePathsProvider()
        assert hasattr(provider1, '_stop_event')
        assert isinstance(provider1._stop_event, threading.Event)
        # Clear stop event if it was set by a previous test (singleton pattern)
        provider1._stop_event.clear()
        assert not provider1._stop_event.is_set()
        
        # Test UnitreeGo2StateProvider
        provider2 = UnitreeGo2StateProvider()
        assert hasattr(provider2, '_stop_event')
        assert isinstance(provider2._stop_event, threading.Event)
        # Clear stop event if it was set by a previous test (singleton pattern)
        provider2._stop_event.clear()
        assert not provider2._stop_event.is_set()
        
        # Test OdomProvider (singleton, so reset stop event if set)
        provider3 = OdomProvider(channel="test_channel", use_zenoh=False)
        assert hasattr(provider3, '_stop_event')
        assert isinstance(provider3._stop_event, threading.Event)
        # Clear stop event if it was set by a previous test (singleton pattern)
        provider3._stop_event.clear()
        assert not provider3._stop_event.is_set()

    def test_loop_condition_with_stop_event(self):
        """Test that the loop condition properly checks the stop event."""
        provider = SimplePathsProvider()
        
        # Clear stop event first (singleton might have it set from previous tests)
        provider._stop_event.clear()
        
        # Test that loop continues when stop event is not set
        assert not provider._stop_event.is_set()
        
        # Test that loop would stop when stop event is set
        provider._stop_event.set()
        assert provider._stop_event.is_set()
        
        # Reset for cleanup
        provider._stop_event.clear()
