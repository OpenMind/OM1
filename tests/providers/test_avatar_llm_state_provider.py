from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from llm.output_model import Action, CortexOutputModel
from providers.avatar_llm_state_provider import AvatarLLMStateProvider


class MockLLM:
    """Mock LLM class for testing"""

    @AvatarLLMStateProvider
    async def ask(self, prompt: str) -> CortexOutputModel:
        """Mock ask method that returns actions"""
        return CortexOutputModel(actions=[Action(type="speak", value="Hello")])

    @AvatarLLMStateProvider
    async def ask_with_face(self, prompt: str) -> CortexOutputModel:
        """Mock ask method that returns face action"""
        return CortexOutputModel(
            actions=[
                Action(type="speak", value="Hello"),
                Action(type="face", value="happy"),
            ]
        )

    @AvatarLLMStateProvider
    async def ask_that_fails(self, prompt: str) -> CortexOutputModel:
        """Mock ask method that raises an exception"""
        raise ValueError("Test error")


@pytest.fixture
def mock_avatar_provider():
    """Fixture to mock AvatarProvider"""
    with patch("providers.avatar_llm_state_provider.AvatarProvider") as mock:
        provider_instance = MagicMock()
        provider_instance.running = True
        provider_instance.send_avatar_command = MagicMock()
        mock.return_value = provider_instance
        yield provider_instance


@pytest.mark.asyncio
async def test_decorator_sets_thinking_state(mock_avatar_provider):
    """Test that decorator sets thinking state before LLM call"""
    llm = MockLLM()
    await llm.ask("test prompt")

    # Verify Think command was sent
    calls = [
        call[0][0] for call in mock_avatar_provider.send_avatar_command.call_args_list
    ]
    assert "Think" in calls


@pytest.mark.asyncio
async def test_decorator_restores_happy_when_no_face_action(mock_avatar_provider):
    """Test that decorator restores happy state when no face action"""
    llm = MockLLM()
    await llm.ask("test prompt")

    # Verify both Think and Happy commands were sent
    calls = [
        call[0][0] for call in mock_avatar_provider.send_avatar_command.call_args_list
    ]
    assert "Think" in calls
    assert "Happy" in calls
    assert calls.index("Think") < calls.index("Happy")


@pytest.mark.asyncio
async def test_decorator_keeps_face_action(mock_avatar_provider):
    """Test that decorator doesn't restore happy when face action exists"""
    llm = MockLLM()
    await llm.ask_with_face("test prompt")

    # Verify Think was called but Happy was not
    calls = [
        call[0][0] for call in mock_avatar_provider.send_avatar_command.call_args_list
    ]
    assert "Think" in calls
    assert "Happy" not in calls


@pytest.mark.asyncio
async def test_decorator_restores_happy_on_exception(mock_avatar_provider):
    """Test that decorator restores happy state when exception occurs"""
    llm = MockLLM()

    with pytest.raises(ValueError, match="Test error"):
        await llm.ask_that_fails("test prompt")

    # Verify both Think and Happy commands were sent
    calls = [
        call[0][0] for call in mock_avatar_provider.send_avatar_command.call_args_list
    ]
    assert "Think" in calls
    assert "Happy" in calls


@pytest.mark.asyncio
async def test_decorator_handles_avatar_provider_not_running():
    """Test that decorator handles when avatar provider is not running"""
    with patch("providers.avatar_llm_state_provider.AvatarProvider") as mock:
        provider_instance = MagicMock()
        provider_instance.running = False
        mock.return_value = provider_instance

        llm = MockLLM()
        result = await llm.ask("test prompt")

        # Should not raise exception
        assert result is not None
        # send_avatar_command should not be called when not running
        provider_instance.send_avatar_command.assert_not_called()


@pytest.mark.asyncio
async def test_decorator_handles_avatar_provider_exception():
    """Test that decorator handles exceptions from avatar provider gracefully"""
    with patch("providers.avatar_llm_state_provider.AvatarProvider") as mock:
        mock.side_effect = Exception("Avatar provider error")

        llm = MockLLM()
        result = await llm.ask("test prompt")

        # Should not raise exception, decorator should catch it
        assert result is not None


@pytest.mark.asyncio
async def test_decorator_preserves_return_value(mock_avatar_provider):
    """Test that decorator preserves the original return value"""
    llm = MockLLM()
    result = await llm.ask("test prompt")

    # Verify the result is what we expect
    assert isinstance(result, CortexOutputModel)
    assert len(result.actions) == 1
    assert result.actions[0].type == "speak"
    assert result.actions[0].value == "Hello"


@pytest.mark.asyncio
async def test_decorator_handles_result_without_actions():
    """Test that decorator handles results without actions attribute"""

    class MockLLMNoActions:
        @AvatarLLMStateProvider
        async def ask(self, prompt: str):
            return {"response": "test"}  # No actions attribute

    with patch("providers.avatar_llm_state_provider.AvatarProvider") as mock:
        provider_instance = MagicMock()
        provider_instance.running = True
        mock.return_value = provider_instance

        llm = MockLLMNoActions()
        result = await llm.ask("test prompt")

        # Should handle gracefully
        assert result is not None


@pytest.mark.asyncio
async def test_decorator_handles_none_result(mock_avatar_provider):
    """Test that decorator handles None result"""

    class MockLLMNone:
        @AvatarLLMStateProvider
        async def ask(self, prompt: str):
            return None

    llm = MockLLMNone()
    result = await llm.ask("test prompt")

    # Should handle gracefully
    assert result is None
    # Happy should still be called since no face action
    calls = [
        call[0][0] for call in mock_avatar_provider.send_avatar_command.call_args_list
    ]
    assert "Think" in calls
    assert "Happy" in calls
