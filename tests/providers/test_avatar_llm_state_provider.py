from unittest.mock import MagicMock, patch

import pytest

from llm.output_model import Action, CortexOutputModel
from providers.avatar_llm_state_provider import AvatarLLMStateProvider


class MockLLM:
    @AvatarLLMStateProvider
    async def ask(self, prompt: str) -> CortexOutputModel:
        return CortexOutputModel(actions=[Action(type="speak", value="Hello")])

    @AvatarLLMStateProvider
    async def ask_with_face(self, prompt: str) -> CortexOutputModel:
        return CortexOutputModel(
            actions=[
                Action(type="speak", value="Hello"),
                Action(type="face", value="happy"),
            ]
        )

    @AvatarLLMStateProvider
    async def ask_that_fails(self, prompt: str) -> CortexOutputModel:
        raise ValueError("Test error")


@pytest.fixture
def mock_avatar_provider() -> MagicMock:
    with patch("providers.avatar_llm_state_provider.AvatarProvider") as mock:
        provider_instance = MagicMock()
        provider_instance.running = True
        provider_instance.send_avatar_command = MagicMock()
        mock.return_value = provider_instance
        yield provider_instance


@pytest.mark.asyncio
async def test_decorator_sets_thinking_state(mock_avatar_provider):
    llm = MockLLM()
    await llm.ask("test prompt")

    calls = [
        args[0] for args, _ in mock_avatar_provider.send_avatar_command.call_args_list
    ]
    assert "Think" in calls


@pytest.mark.asyncio
async def test_decorator_restores_happy_when_no_face_action(mock_avatar_provider):
    llm = MockLLM()
    await llm.ask("test prompt")

    calls = [
        args[0] for args, _ in mock_avatar_provider.send_avatar_command.call_args_list
    ]
    assert "Think" in calls
    assert "Happy" in calls
    assert calls.index("Think") < calls.index("Happy")


@pytest.mark.asyncio
async def test_decorator_keeps_face_action(mock_avatar_provider):
    llm = MockLLM()
    await llm.ask_with_face("test prompt")

    calls = [
        args[0] for args, _ in mock_avatar_provider.send_avatar_command.call_args_list
    ]
    assert "Think" in calls
    assert "Happy" not in calls


@pytest.mark.asyncio
async def test_decorator_restores_happy_on_exception(mock_avatar_provider):
    llm = MockLLM()
    with pytest.raises(ValueError, match="Test error"):
        await llm.ask_that_fails("test prompt")

    calls = [
        args[0] for args, _ in mock_avatar_provider.send_avatar_command.call_args_list
    ]
    assert "Think" in calls
    assert "Happy" in calls


@pytest.mark.asyncio
async def test_decorator_handles_avatar_provider_not_running():
    with patch("providers.avatar_llm_state_provider.AvatarProvider") as mock:
        provider_instance = MagicMock()
        provider_instance.running = False
        provider_instance.send_avatar_command = MagicMock()
        mock.return_value = provider_instance

        llm = MockLLM()
        result = await llm.ask("test prompt")

        assert result is not None
        provider_instance.send_avatar_command.assert_not_called()


@pytest.mark.asyncio
async def test_decorator_handles_avatar_provider_exception():
    with patch("providers.avatar_llm_state_provider.AvatarProvider") as mock:
        mock.side_effect = Exception("Avatar provider error")

        llm = MockLLM()
        result = await llm.ask("test prompt")

        assert result is not None


@pytest.mark.asyncio
async def test_decorator_preserves_return_value(mock_avatar_provider):
    llm = MockLLM()
    result = await llm.ask("test prompt")

    assert isinstance(result, CortexOutputModel)
    assert len(result.actions) == 1
    assert result.actions[0].type == "speak"
    assert result.actions[0].value == "Hello"


@pytest.mark.asyncio
async def test_decorator_handles_result_without_actions(mock_avatar_provider):
    class MockLLMNoActions:
        @AvatarLLMStateProvider
        async def ask(self, prompt: str):
            return {"response": "test"}

    llm = MockLLMNoActions()
    result = await llm.ask("test prompt")

    assert result is not None
    calls = [
        args[0] for args, _ in mock_avatar_provider.send_avatar_command.call_args_list
    ]
    assert "Think" in calls
    assert "Happy" in calls


@pytest.mark.asyncio
async def test_decorator_handles_none_result(mock_avatar_provider):
    class MockLLMNone:
        @AvatarLLMStateProvider
        async def ask(self, prompt: str):
            return None

    llm = MockLLMNone()
    result = await llm.ask("test prompt")

    assert result is None
    calls = [
        args[0] for args, _ in mock_avatar_provider.send_avatar_command.call_args_list
    ]
    assert "Think" in calls
    assert "Happy" in calls
