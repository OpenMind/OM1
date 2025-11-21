from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from llm.output_model import Action, CortexOutputModel
from providers.avatar_llm_state_provider import AvatarLLMState


class MockLLM:
    @AvatarLLMState.trigger_thinking
    async def ask(self, prompt: str) -> CortexOutputModel:
        return CortexOutputModel(actions=[Action(type="speak", value="Hello")])

    @AvatarLLMState.trigger_thinking
    async def ask_with_face(self, prompt: str) -> CortexOutputModel:
        return CortexOutputModel(
            actions=[
                Action(type="speak", value="Hello"),
                Action(type="face", value="happy"),
            ]
        )

    @AvatarLLMState.trigger_thinking
    async def ask_that_fails(self, prompt: str) -> CortexOutputModel:
        raise ValueError("Test error")


def reset_avatar_llm_state():
    AvatarLLMState._instance = None


@pytest.fixture
def mock_avatar_provider() -> Generator[MagicMock, None, None]:
    reset_avatar_llm_state()

    with (
        patch("providers.avatar_llm_state_provider.AvatarProvider") as avatar_mock,
        patch("providers.avatar_llm_state_provider.IOProvider") as io_mock,
    ):

        provider_instance = MagicMock()
        provider_instance.running = True
        provider_instance.send_avatar_command = MagicMock()
        provider_instance.stop = MagicMock()
        avatar_mock.return_value = provider_instance

        io_instance = MagicMock()
        io_instance.llm_prompt = "INPUT: Voice\ntest prompt"
        io_mock.return_value = io_instance

        yield provider_instance

    reset_avatar_llm_state()


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
    reset_avatar_llm_state()

    with (
        patch("providers.avatar_llm_state_provider.AvatarProvider") as avatar_mock,
        patch("providers.avatar_llm_state_provider.IOProvider") as io_mock,
    ):

        provider_instance = MagicMock()
        provider_instance.running = False
        provider_instance.send_avatar_command = MagicMock()
        provider_instance.stop = MagicMock()

        avatar_mock.return_value = provider_instance

        io_instance = MagicMock()
        io_instance.llm_prompt = "INPUT: Voice\ntest prompt"
        io_mock.return_value = io_instance

        llm = MockLLM()
        result = await llm.ask("test prompt")

        assert result is not None
        provider_instance.send_avatar_command.assert_not_called()

    reset_avatar_llm_state()


@pytest.mark.asyncio
async def test_decorator_handles_avatar_provider_exception():
    reset_avatar_llm_state()

    with (
        patch("providers.avatar_llm_state_provider.AvatarProvider") as avatar_mock,
        patch("providers.avatar_llm_state_provider.IOProvider") as io_mock,
    ):

        avatar_mock.side_effect = Exception("Avatar provider error")

        io_instance = MagicMock()
        io_instance.llm_prompt = "INPUT: Voice\ntest prompt"
        io_mock.return_value = io_instance

        llm = MockLLM()
        result = await llm.ask("test prompt")

        assert result is not None

    reset_avatar_llm_state()


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
        @AvatarLLMState.trigger_thinking
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
        @AvatarLLMState.trigger_thinking
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
