import types
from unittest.mock import mock_open, patch

import pytest
from pydantic import BaseModel

from llm import LLM, LLMConfig, find_module_with_class, load_llm
from providers.io_provider import IOProvider
from runtime.single_mode.config import add_meta


class DummyOutputModel(BaseModel):
    test_field: str


class MockLLM(LLM[BaseModel]):
    async def ask(self, prompt: str, messages=None) -> BaseModel:
        raise NotImplementedError


@pytest.fixture
def config():
    return LLMConfig(base_url="test_url", api_key="test_key", model="test_model")


@pytest.fixture
def base_llm(config):
    return MockLLM(config, available_actions=None)


def test_llm_init(base_llm, config):
    assert base_llm._config == config
    assert isinstance(base_llm.io_provider, type(IOProvider()))


@pytest.mark.asyncio
async def test_llm_ask_not_implemented(base_llm):
    with pytest.raises(NotImplementedError):
        await base_llm.ask("test prompt")


def test_llm_config():
    llm_config = LLMConfig(
        **add_meta(  # type: ignore
            {
                "config_key": "config_value",
            },
            None,
            None,
            None,
            None,
        )
    )
    assert llm_config.config_key == "config_value"  # type: ignore
    with pytest.raises(
        AttributeError, match="'LLMConfig' object has no attribute 'invalid_key'"
    ):
        llm_config.invalid_key  # type: ignore


def test_load_llm_mock_implementation():

    with (
        patch("llm.find_module_with_class") as mock_find_module,
        patch("llm.importlib.import_module") as mock_import,
    ):
        mock_find_module.return_value = "mock_llm"
        mock_module = types.ModuleType("mock_llm")
        setattr(mock_module, "MockLLM", MockLLM)
        mock_import.return_value = mock_module

        result = load_llm({"type": "MockLLM"})

        mock_find_module.assert_called_once_with("MockLLM")
        mock_import.assert_called_once_with("llm.plugins.mock_llm")
        assert isinstance(result, LLM)


def test_load_llm_not_found():
    with patch("llm.find_module_with_class") as mock_find_module:
        mock_find_module.return_value = None

        with pytest.raises(
            ValueError,
            match="Class 'NonexistentLLM' not found in .*LLM plugin module",
        ):
            load_llm({"type": "NonexistentLLM"})


def test_load_llm_invalid_type():

    with (
        patch("llm.find_module_with_class") as mock_find_module,
        patch("llm.importlib.import_module") as mock_import,
    ):
        mock_find_module.return_value = "invalid_llm"

        class InvalidLLM:
            pass

        mock_module = types.ModuleType("invalid_llm")
        setattr(mock_module, "InvalidLLM", InvalidLLM)
        mock_import.return_value = mock_module

        with pytest.raises(
            ValueError, match="'InvalidLLM' is not a valid LLM subclass"
        ):
            load_llm({"type": "InvalidLLM"})


def test_find_module_with_class_success():
    with (
        patch("os.path.join") as mock_join,
        patch("os.path.exists") as mock_exists,
        patch("os.listdir") as mock_listdir,
        patch("builtins.open", mock_open(read_data="class TestLLM(LLM):\n    pass\n")),
    ):
        mock_join.side_effect = lambda *args: "/".join(args)
        mock_exists.return_value = True
        mock_listdir.return_value = ["test_llm.py"]

        result = find_module_with_class("TestLLM")

        assert result == "test_llm"


def test_find_module_with_class_not_found():
    with (
        patch("os.path.join") as mock_join,
        patch("os.path.exists") as mock_exists,
        patch("os.listdir") as mock_listdir,
        patch("builtins.open", mock_open(read_data="class OtherClass:\n    pass\n")),
    ):
        mock_join.side_effect = lambda *args: "/".join(args)
        mock_exists.return_value = True
        mock_listdir.return_value = ["other_file.py"]

        result = find_module_with_class("TestLLM")

        assert result is None


def test_find_module_with_class_no_plugins_dir():
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False

        result = find_module_with_class("TestLLM")

        assert result is None
