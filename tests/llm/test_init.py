from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from llm import LLM, LLMConfig, load_llm
from providers.io_provider import IOProvider
from runtime.config import add_meta


class DummyOutputModel(BaseModel):
    test_field: str


class MockLLM(LLM[BaseModel]):
    async def ask(self, prompt: str) -> BaseModel:
        raise NotImplementedError


@pytest.fixture
def config():
    return LLMConfig(base_url="test_url", api_key="test_key", model="test_model")


@pytest.fixture
def base_llm(config):
    return MockLLM(DummyOutputModel, config)


def test_llm_init(base_llm, config):
    assert base_llm._output_model == DummyOutputModel
    assert base_llm._config == config
    assert isinstance(base_llm.io_provider, type(IOProvider()))


def test_llm_init_no_config():
    llm = MockLLM(DummyOutputModel, None)
    assert llm._config is None


@pytest.mark.asyncio
async def test_llm_ask_not_implemented(base_llm):
    with pytest.raises(NotImplementedError):
        await base_llm.ask("test prompt")


def test_llm_config():
    llm_config = LLMConfig(
        **add_meta(
            {
                "config_key": "config_value",
            },
            None,
            None,
            None,
            None,
        )
    )
    assert llm_config.config_key == "config_value"
    with pytest.raises(
        AttributeError, match="'LLMConfig' object has no attribute 'invalid_key'"
    ):
        llm_config.invalid_key


def test_load_llm_mock_implementation():
    with (
        patch("os.path.dirname") as mock_dirname,
        patch("os.listdir") as mock_listdir,
        patch("importlib.import_module") as mock_import,
    ):

        mock_dirname.return_value = "/test/dir"
        mock_listdir.return_value = ["mock_llm.py"]

        mock_module = Mock()
        mock_module.MockLLM = MockLLM
        mock_import.return_value = mock_module

        result = load_llm("MockLLM")

        mock_import.assert_called_once_with("llm.plugins.mock_llm")
        assert result == MockLLM


def test_load_llm_not_found():
    with patch("os.path.dirname") as mock_dirname, patch("os.listdir") as mock_listdir:
        mock_dirname.return_value = "/test/dir"
        mock_listdir.return_value = []

        with pytest.raises(ValueError, match="LLM type NonexistentLLM not found"):
            load_llm("NonexistentLLM")
