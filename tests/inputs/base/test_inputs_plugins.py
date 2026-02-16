import asyncio
from typing import Type

import pytest

from inputs.base.loop import FuserInput


def get_all_inputs_classes():
    import importlib
    import inspect
    import logging
    import os

    plugins_dir = os.path.join("src", "inputs", "plugins")
    plugin_files = [f[:-3] for f in os.listdir(plugins_dir) if f.endswith(".py")]

    inputs_classes = []
    for plugin in plugin_files:
        try:
            module = importlib.import_module(f"inputs.plugins.{plugin}")
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, FuserInput)
                    and obj != FuserInput
                ):
                    inputs_classes.append(obj)
        except (ImportError, ModuleNotFoundError) as e:
            # Skip plugins that fail to import due to missing optional dependencies
            logging.warning(f"Skipping plugin {plugin} due to import error: {e}")
            continue
    return inputs_classes


@pytest.mark.parametrize("input_class", get_all_inputs_classes())
def test_init_signature(input_class: Type[FuserInput]):
    # Verify __init__ signature matches base class
    base_params = set(FuserInput.__init__.__annotations__.keys())
    impl_params = set(input_class.__init__.__annotations__.keys())
    assert (
        base_params == impl_params
    ), f"{input_class.__name__} __init__ signature mismatch"


@pytest.mark.parametrize("input_class", get_all_inputs_classes())
def test__poll_to_text_signature(input_class: Type[FuserInput]):
    # Verify _poll method signature matches base class
    base_params = set(FuserInput._poll.__annotations__.keys())
    impl_params = set(input_class._poll.__annotations__.keys())
    assert (
        base_params == impl_params
    ), f"{input_class.__name__} _poll signature mismatch"


@pytest.mark.parametrize("input_class", get_all_inputs_classes())
def test__listen_loop_to_text_signature(input_class: Type[FuserInput]):
    # Verify _listen_loop method signature matches base class
    base_params = set(FuserInput._listen_loop.__annotations__.keys())
    impl_params = set(input_class._listen_loop.__annotations__.keys())
    assert (
        base_params == impl_params
    ), f"{input_class.__name__} _listen_loop signature mismatch"


@pytest.mark.parametrize("input_class", get_all_inputs_classes())
def test__raw_to_text_signature(input_class: Type[FuserInput]):
    # Verify _raw_to_text method signature matches base class
    base_params = set(FuserInput._raw_to_text.__annotations__.keys())
    impl_params = set(input_class._raw_to_text.__annotations__.keys())
    assert (
        base_params == impl_params
    ), f"{input_class.__name__} _raw_to_text signature mismatch"


@pytest.mark.parametrize("input_class", get_all_inputs_classes())
def test_raw_to_text_signature(input_class: Type[FuserInput]):
    # Verify _raw_to_text method signature matches base class
    base_params = set(FuserInput.raw_to_text.__annotations__.keys())
    impl_params = set(input_class.raw_to_text.__annotations__.keys())
    assert (
        base_params == impl_params
    ), f"{input_class.__name__} raw_to_text signature mismatch"


@pytest.mark.parametrize("input_class", get_all_inputs_classes())
def test_formatted_latest_buffer_signature(input_class: Type[FuserInput]):
    # Verify formatted_latest_buffer method signature matches base class
    base_params = set(FuserInput.formatted_latest_buffer.__annotations__.keys())
    impl_params = set(input_class.formatted_latest_buffer.__annotations__.keys())
    assert (
        base_params == impl_params
    ), f"{input_class.__name__} formatted_latest_buffer signature mismatch"


@pytest.mark.parametrize("input_class", get_all_inputs_classes())
def test_listen_signature(input_class: Type[FuserInput]):
    # Verify listen method signature matches base class
    base_params = set(FuserInput.listen.__annotations__.keys())
    impl_params = set(input_class.listen.__annotations__.keys())
    assert (
        base_params == impl_params
    ), f"{input_class.__name__} listen signature mismatch"


class TestInputBaseInternals:
    """Test internal methods for coverage."""

    @pytest.mark.asyncio
    async def test_fuser_input_super_init(self):
        """Test FuserInput calls super().__init__ (line 26)."""
        from inputs.base import SensorConfig
        from inputs.base.loop import FuserInput

        class TestConfig(SensorConfig):
            test_field: str = "test"

        class TestFuser(FuserInput[TestConfig, str]):
            async def _poll(self):
                return "data"

        config = TestConfig(test_field="value")
        fuser = TestFuser(config)
        assert fuser.config == config
        assert fuser.config.test_field == "value"

    @pytest.mark.asyncio
    async def test_fuser_input_listen_loop_polling(self):
        """Test FuserInput._listen_loop polling (lines 37-38)."""
        from inputs.base import SensorConfig
        from inputs.base.loop import FuserInput

        class TestConfig(SensorConfig):
            pass

        class TestFuser(FuserInput[TestConfig, str]):
            def __init__(self, config):
                super().__init__(config)
                self.call_count = 0

            async def _poll(self):
                self.call_count += 1
                await asyncio.sleep(0.001)
                return f"poll_{self.call_count}"

        fuser = TestFuser(TestConfig())
        results = []
        async for item in fuser._listen_loop():
            results.append(item)
            if len(results) >= 3:
                break

        assert results == ["poll_1", "poll_2", "poll_3"]
        assert fuser.call_count == 3


class TestSensorListenMethod:
    """Test Sensor._listen to cover lines 127-128."""

    @pytest.mark.asyncio
    async def test_sensor_listen_iterates_listen_loop(self):
        """Test Sensor._listen properly iterates over _listen_loop (lines 127-128)."""
        from inputs.base import Sensor, SensorConfig

        class TestConfig(SensorConfig):
            pass

        class TestSensor(Sensor[TestConfig, str]):
            def __init__(self, config):
                super().__init__(config)
                self.events = ["event1", "event2", "event3"]

            async def _listen_loop(self):
                for event in self.events:
                    yield event

        sensor = TestSensor(TestConfig())
        collected = []
        async for event in sensor.listen():
            collected.append(event)

        assert collected == ["event1", "event2", "event3"]
