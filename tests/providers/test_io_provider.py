import time
from typing import Optional

import pytest

from providers.io_provider import Input, IOProvider


@pytest.fixture
def io_provider():
    provider = IOProvider()
    yield provider
    provider._inputs.clear()
    provider._fuser_start_time = None
    provider._fuser_end_time = None
    provider._llm_prompt = None
    provider._llm_start_time = None
    provider._llm_end_time = None
    provider._tick_counter = 0


def test_add_input_with_timestamp(io_provider):
    timestamp = time.time()
    io_provider.add_input("key1", "value1", timestamp)
    assert io_provider.inputs["key1"] == Input(
        input="value1", timestamp=timestamp, tick=0
    )


def test_remove_input(io_provider):
    io_provider.add_input("key1", "value1", None)
    io_provider.remove_input("key1")
    assert "key1" not in io_provider.inputs


def test_add_input_timestamp(io_provider):
    timestamp = time.time()
    io_provider.add_input("key1", "value1", None)
    io_provider.add_input_timestamp("key1", timestamp)
    assert io_provider.get_input_timestamp("key1") == timestamp


def test_get_input_timestamp_nonexistent_key(io_provider):
    assert io_provider.get_input_timestamp("nonexistent") is None


def test_fuser_time_properties(io_provider):
    start_time = time.time()
    end_time = start_time + 1.0

    io_provider.fuser_start_time = start_time
    assert io_provider.fuser_start_time == start_time

    io_provider.fuser_end_time = end_time
    assert io_provider.fuser_end_time == end_time


def test_llm_properties(io_provider):
    prompt = "test prompt"
    start_time = time.time()
    end_time = start_time + 1.0

    io_provider.llm_prompt = prompt
    assert io_provider.llm_prompt == prompt

    io_provider.llm_start_time = start_time
    assert io_provider.llm_start_time == start_time

    io_provider.llm_end_time = end_time
    assert io_provider.llm_end_time == end_time


def test_clear_llm_prompt(io_provider):
    io_provider.llm_prompt = "test prompt"
    io_provider.clear_llm_prompt()
    assert io_provider.llm_prompt is None


def test_singleton_behavior():
    provider1 = IOProvider()
    provider2 = IOProvider()
    assert provider1 is provider2


def test_thread_safety(io_provider):
    import threading
    import time

    def worker(key: str, value: str, timestamp: Optional[float]):
        io_provider.add_input(key, value, timestamp)

    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(f"key{i}", f"value{i}", time.time()))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(io_provider.inputs) == 10
