import time
from typing import Optional

import pytest

from providers.io_provider import Input, IOProvider


@pytest.fixture
def io_provider():
    provider = IOProvider()
    yield provider
    provider._inputs.clear()
    provider._fuser_system_prompt = None
    provider._fuser_inputs = None
    provider._fuser_available_actions = None
    provider._fuser_start_time = None
    provider._fuser_end_time = None
    provider._llm_prompt = None
    provider._llm_start_time = None
    provider._llm_end_time = None
    provider._mode_transition_input = None
    provider._variables.clear()
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


def test_add_input_without_timestamp(io_provider):
    io_provider.add_input("key1", "value1", None)
    input_obj = io_provider.get_input("key1")
    assert input_obj is not None
    assert input_obj.input == "value1"
    assert input_obj.timestamp is not None
    assert input_obj.tick == 0


def test_get_input(io_provider):
    io_provider.add_input("key1", "value1", None)
    input_obj = io_provider.get_input("key1")
    assert input_obj is not None
    assert input_obj.input == "value1"


def test_get_input_nonexistent(io_provider):
    assert io_provider.get_input("nonexistent") is None


def test_fuser_system_prompt_property(io_provider):
    prompt = "system prompt"
    io_provider.fuser_system_prompt = prompt
    assert io_provider.fuser_system_prompt == prompt


def test_set_fuser_system_prompt_method(io_provider):
    prompt = "system prompt"
    io_provider.set_fuser_system_prompt(prompt)
    assert io_provider.fuser_system_prompt == prompt


def test_fuser_inputs_property(io_provider):
    inputs = "fuser inputs"
    io_provider.fuser_inputs = inputs
    assert io_provider.fuser_inputs == inputs


def test_set_fuser_inputs_method(io_provider):
    inputs = "fuser inputs"
    io_provider.set_fuser_inputs(inputs)
    assert io_provider.fuser_inputs == inputs


def test_fuser_available_actions_property(io_provider):
    actions = "available actions"
    io_provider.fuser_available_actions = actions
    assert io_provider.fuser_available_actions == actions


def test_set_fuser_available_actions_method(io_provider):
    actions = "available actions"
    io_provider.set_fuser_available_actions(actions)
    assert io_provider.fuser_available_actions == actions


def test_set_fuser_start_time_method(io_provider):
    start_time = time.time()
    io_provider.set_fuser_start_time(start_time)
    assert io_provider.fuser_start_time == start_time


def test_set_fuser_end_time_method(io_provider):
    end_time = time.time()
    io_provider.set_fuser_end_time(end_time)
    assert io_provider.fuser_end_time == end_time


def test_set_llm_prompt_method(io_provider):
    prompt = "llm prompt"
    io_provider.set_llm_prompt(prompt)
    assert io_provider.llm_prompt == prompt


def test_set_llm_start_time_method(io_provider):
    start_time = time.time()
    io_provider.set_llm_start_time(start_time)
    assert io_provider.llm_start_time == start_time


def test_dynamic_variables(io_provider):
    io_provider.add_dynamic_variable("var1", "value1")
    io_provider.add_dynamic_variable("var2", 42)
    io_provider.add_dynamic_variable("var3", {"nested": "dict"})

    assert io_provider.get_dynamic_variable("var1") == "value1"
    assert io_provider.get_dynamic_variable("var2") == 42
    assert io_provider.get_dynamic_variable("var3") == {"nested": "dict"}


def test_get_dynamic_variable_nonexistent(io_provider):
    assert io_provider.get_dynamic_variable("nonexistent") is None


def test_add_mode_transition_input(io_provider):
    io_provider.add_mode_transition_input("first input")
    assert io_provider.get_mode_transition_input() == "first input"

    io_provider.add_mode_transition_input("second input")
    assert io_provider.get_mode_transition_input() == "first input second input"


def test_delete_mode_transition_input(io_provider):
    io_provider.add_mode_transition_input("test input")
    assert io_provider.get_mode_transition_input() == "test input"

    io_provider.delete_mode_transition_input()
    assert io_provider.get_mode_transition_input() is None


def test_mode_transition_input_context_manager(io_provider):
    io_provider.add_mode_transition_input("context input")

    with io_provider.mode_transition_input() as input_text:
        assert input_text == "context input"

    assert io_provider.get_mode_transition_input() is None


def test_mode_transition_input_context_manager_empty(io_provider):
    with io_provider.mode_transition_input() as input_text:
        assert input_text is None

    assert io_provider.get_mode_transition_input() is None


def test_tick_counter_property(io_provider):
    assert io_provider.tick_counter == 0


def test_increment_tick(io_provider):
    assert io_provider.increment_tick() == 1
    assert io_provider.increment_tick() == 2
    assert io_provider.tick_counter == 2


def test_reset_tick_counter(io_provider):
    io_provider.increment_tick()
    io_provider.increment_tick()
    assert io_provider.tick_counter == 2

    io_provider.reset_tick_counter()
    assert io_provider.tick_counter == 0


def test_tick_tracking_in_inputs(io_provider):
    io_provider.reset_tick_counter()

    io_provider.add_input("key1", "value1", None)
    assert io_provider.get_input("key1").tick == 0

    io_provider.increment_tick()
    io_provider.add_input("key2", "value2", None)
    assert io_provider.get_input("key2").tick == 1

    io_provider.increment_tick()
    io_provider.add_input("key3", "value3", None)
    assert io_provider.get_input("key3").tick == 2


def test_add_input_timestamp_nonexistent_key(io_provider):
    io_provider.add_input_timestamp("nonexistent", time.time())
    assert io_provider.get_input("nonexistent") is None


def test_inputs_property_returns_copy(io_provider):
    io_provider.add_input("key1", "value1", None)
    inputs1 = io_provider.inputs
    inputs2 = io_provider.inputs

    assert inputs1 is not inputs2
    assert inputs1 == inputs2
