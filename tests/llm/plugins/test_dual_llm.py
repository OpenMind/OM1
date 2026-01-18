import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm.output_model import Action, CortexOutputModel
from llm.plugins.dual_llm import DualLLM, DualLLMConfig, _extract_voice_input


@pytest.fixture(autouse=True)
def mock_decorators():
    """Mock decorators used in DualLLM"""

    def mock_decorator(func=None):
        def decorator(f):
            return f

        if func is not None:
            return decorator(func)
        return decorator

    with (
        patch("llm.plugins.dual_llm.AvatarLLMState.trigger_thinking", mock_decorator),
        patch("llm.plugins.dual_llm.LLMHistoryManager.update_history", mock_decorator),
        patch("llm.plugins.dual_llm.LLMHistoryManager"),
    ):
        yield


@pytest.fixture
def mock_llm_classes():
    """Mock get_llm_class to return mock LLM classes"""
    local_llm_mock = MagicMock()
    local_llm_instance = MagicMock()
    local_llm_instance.ask = AsyncMock()
    local_llm_mock.return_value = local_llm_instance

    cloud_llm_mock = MagicMock()
    cloud_llm_instance = MagicMock()
    cloud_llm_instance.ask = AsyncMock()
    cloud_llm_mock.return_value = cloud_llm_instance

    with patch("llm.plugins.dual_llm.get_llm_class") as mock_get_class:

        def get_class_side_effect(name):
            if name == "MockLocal":
                return local_llm_mock
            elif name == "MockCloud":
                return cloud_llm_mock
            return MagicMock()

        mock_get_class.side_effect = get_class_side_effect
        yield local_llm_instance, cloud_llm_instance


@pytest.fixture
def dual_llm(mock_llm_classes):
    config = DualLLMConfig(
        local_llm_type="MockLocal", cloud_llm_type="MockCloud", api_key="test_key"
    )
    with patch("openai.AsyncClient"):
        llm = DualLLM(config)

    llm._local_llm = mock_llm_classes[0]
    llm._cloud_llm = mock_llm_classes[1]

    llm.io_provider = MagicMock()

    return llm


def test_extract_voice_input():
    assert (
        _extract_voice_input("INPUT: Voice // START create a folder // END")
        == "create a folder"
    )
    assert _extract_voice_input("Normal prompt") == ""
    assert (
        _extract_voice_input("INPUT: Voice // START   multi line \n input  // END")
        == "multi line \n input"
    )


@pytest.mark.asyncio
async def test_race_local_wins(dual_llm):
    """Test scenario where local LLM replies first"""
    local_result = CortexOutputModel(actions=[])
    dual_llm._local_llm.ask.return_value = local_result

    async def fast_local(*args):
        return local_result

    async def slow_cloud(*args):
        await asyncio.sleep(0.5)
        return CortexOutputModel(actions=[])

    dual_llm._local_llm.ask = AsyncMock(side_effect=fast_local)
    dual_llm._cloud_llm.ask = AsyncMock(side_effect=slow_cloud)

    dual_llm.TIMEOUT_THRESHOLD = 0.1

    response = await dual_llm.ask("test prompt")

    assert response == local_result
    assert dual_llm._local_llm.ask.called
    assert dual_llm._cloud_llm.ask.called


@pytest.mark.asyncio
async def test_race_cloud_wins(dual_llm):
    """Test scenario where cloud LLM replies first"""
    cloud_result = CortexOutputModel(actions=[])

    async def slow_local(*args):
        await asyncio.sleep(0.5)
        return CortexOutputModel(actions=[])

    async def fast_cloud(*args):
        return cloud_result

    dual_llm._local_llm.ask = AsyncMock(side_effect=slow_local)
    dual_llm._cloud_llm.ask = AsyncMock(side_effect=fast_cloud)

    dual_llm.TIMEOUT_THRESHOLD = 0.1

    response = await dual_llm.ask("test prompt")

    assert response == cloud_result


@pytest.mark.asyncio
async def test_race_both_fast_local_has_function(dual_llm):
    """Test scenario: Both fast, Local has function call -> Local wins"""
    local_result = CortexOutputModel(actions=[Action(type="func", value="val")])
    cloud_result = CortexOutputModel(actions=[])

    async def fast(*args):
        return None

    dual_llm._local_llm.ask = AsyncMock(return_value=local_result)
    dual_llm._cloud_llm.ask = AsyncMock(return_value=cloud_result)

    dual_llm.TIMEOUT_THRESHOLD = 1.0

    response = await dual_llm.ask("test prompt")

    assert response == local_result


@pytest.mark.asyncio
async def test_race_both_fast_cloud_has_function(dual_llm):
    """Test scenario: Both fast, Cloud has function call -> Cloud wins"""
    local_result = CortexOutputModel(actions=[])
    cloud_result = CortexOutputModel(actions=[Action(type="func", value="val")])

    dual_llm._local_llm.ask = AsyncMock(return_value=local_result)
    dual_llm._cloud_llm.ask = AsyncMock(return_value=cloud_result)
    dual_llm.TIMEOUT_THRESHOLD = 1.0

    response = await dual_llm.ask("test prompt")
    assert response == cloud_result


@pytest.mark.asyncio
async def test_race_both_fast_both_functions_eval(dual_llm):
    """Test scenario: Both fast and have functions -> LLM Judge decides"""
    local_result = CortexOutputModel(actions=[Action(type="f", value="v")])
    cloud_result = CortexOutputModel(actions=[Action(type="f", value="v")])

    dual_llm._local_llm.ask = AsyncMock(return_value=local_result)
    dual_llm._cloud_llm.ask = AsyncMock(return_value=cloud_result)
    dual_llm.TIMEOUT_THRESHOLD = 1.0

    mock_eval_response = MagicMock()
    mock_eval_response.choices = [MagicMock(message=MagicMock(content="B"))]
    dual_llm._eval_client.chat.completions.create = AsyncMock(
        return_value=mock_eval_response
    )

    response = await dual_llm.ask("test prompt")

    assert response == cloud_result
    assert dual_llm._eval_client.chat.completions.create.called


@pytest.mark.asyncio
async def test_timeout_both_slow(dual_llm):
    """Test scenario: Both slow -> Wait for first to complete"""
    local_result = CortexOutputModel(actions=[])
    cloud_result = CortexOutputModel(actions=[])

    async def delayed_local(*args):
        await asyncio.sleep(0.2)
        return local_result

    async def delayed_cloud(*args):
        await asyncio.sleep(0.3)
        return cloud_result

    dual_llm._local_llm.ask = AsyncMock(side_effect=delayed_local)
    dual_llm._cloud_llm.ask = AsyncMock(side_effect=delayed_cloud)
    dual_llm.TIMEOUT_THRESHOLD = 0.1

    response = await dual_llm.ask("test prompt")

    assert response == local_result
