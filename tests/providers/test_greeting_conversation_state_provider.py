import time
from unittest.mock import MagicMock

import pytest

from providers.greeting_conversation_state_provider import (
    ConfidenceCalculator,
    ConfidenceFactors,
    ConversationState,
    GreetingConversationStateMachineProvider,
)
from providers.io_provider import Input


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton instance before each test to ensure test isolation."""
    GreetingConversationStateMachineProvider.reset()  # type: ignore
    yield
    GreetingConversationStateMachineProvider.reset()  # type: ignore


@pytest.fixture
def confidence_calculator():
    """Fixture for ConfidenceCalculator."""
    return ConfidenceCalculator()


@pytest.fixture
def state_machine():
    """Fixture for GreetingConversationStateMachineProvider."""
    provider = GreetingConversationStateMachineProvider()

    provider.reset_state()
    yield provider

    provider.reset_state()


@pytest.fixture
def state_machine_with_mock_io():
    """Fixture for GreetingConversationStateMachineProvider with mocked IOProvider."""
    provider = GreetingConversationStateMachineProvider()
    provider.reset_state()

    mock_io = MagicMock()
    mock_io.tick_counter = 0
    mock_io.get_input.return_value = None
    provider.io_provider = mock_io

    yield provider, mock_io

    provider.reset_state()


class TestConfidenceCalculator:
    """Test suite for ConfidenceCalculator."""

    def test_calculate_completion_confidence_basic(self, confidence_calculator):
        """Test basic confidence calculation."""
        factors = ConfidenceFactors(
            conversation_state=ConversationState.CONVERSING,
            llm_confidence=0.8,
            silence_duration=1.0,
            speech_clarity=0.9,
            person_distance=1.5,
            conversation_duration=30.0,
            turn_count=5,
            last_user_utterance_length=10,
        )

        result = confidence_calculator.calculate_completion_confidence(factors)

        assert "overall" in result
        assert "breakdown" in result
        assert "factors" in result
        assert 0.0 <= result["overall"] <= 1.0
        assert result["breakdown"]["llm"] >= 0.0
        assert result["breakdown"]["silence"] >= 0.0
        assert result["breakdown"]["engagement"] >= 0.0
        assert result["breakdown"]["quality"] >= 0.0

    def test_calculate_confidence_high_silence(self, confidence_calculator):
        """Test confidence with high silence duration."""
        factors = ConfidenceFactors(
            conversation_state=ConversationState.CONVERSING,
            llm_confidence=0.5,
            silence_duration=8.0,  # Long silence
            speech_clarity=0.9,
            person_distance=1.5,
            conversation_duration=30.0,
            turn_count=3,
            last_user_utterance_length=5,
        )

        result = confidence_calculator.calculate_completion_confidence(factors)

        # High silence should increase completion confidence
        assert result["breakdown"]["silence"] > 0.5

    def test_calculate_confidence_person_leaving(self, confidence_calculator):
        """Test confidence when person is far away."""
        factors = ConfidenceFactors(
            conversation_state=ConversationState.CONVERSING,
            llm_confidence=0.5,
            silence_duration=2.0,
            speech_clarity=0.9,
            person_distance=4.0,  # Far away
            conversation_duration=20.0,
            turn_count=2,
            last_user_utterance_length=3,
        )

        result = confidence_calculator.calculate_completion_confidence(factors)

        # Far distance should increase completion confidence
        assert result["breakdown"]["engagement"] > 0.5

    def test_calculate_confidence_short_responses(self, confidence_calculator):
        """Test confidence with very short user responses."""
        factors = ConfidenceFactors(
            conversation_state=ConversationState.CONVERSING,
            llm_confidence=0.5,
            silence_duration=2.0,
            speech_clarity=0.9,
            person_distance=1.5,
            conversation_duration=25.0,
            turn_count=5,
            last_user_utterance_length=2,  # Very short response
        )

        result = confidence_calculator.calculate_completion_confidence(factors)

        # Short response should increase completion confidence
        assert result["breakdown"]["engagement"] > 0.0

    def test_should_transition_to_concluding_high_confidence(
        self, confidence_calculator
    ):
        """Test transition to concluding with high overall confidence."""
        factors = ConfidenceFactors(
            conversation_state=ConversationState.CONVERSING,
            llm_confidence=0.9,
            silence_duration=5.0,
            speech_clarity=0.9,
            person_distance=3.0,
            conversation_duration=30.0,
            turn_count=5,
            last_user_utterance_length=3,
        )

        result = confidence_calculator.calculate_completion_confidence(factors)
        should_transition = confidence_calculator.should_transition_to_concluding(
            result
        )

        # High confidence should trigger transition
        assert should_transition is True

    def test_should_transition_to_concluding_llm_wants(self, confidence_calculator):
        """Test transition when LLM explicitly wants to conclude."""
        factors = ConfidenceFactors(
            conversation_state=ConversationState.CONCLUDING,  # LLM says concluding
            llm_confidence=0.8,
            silence_duration=2.0,
            speech_clarity=0.9,
            person_distance=1.5,
            conversation_duration=20.0,
            turn_count=3,
            last_user_utterance_length=5,
        )

        result = confidence_calculator.calculate_completion_confidence(factors)
        should_transition = confidence_calculator.should_transition_to_concluding(
            result
        )

        # LLM explicitly wants to conclude with decent confidence
        assert should_transition is True

    def test_should_not_transition_to_concluding_low_confidence(
        self, confidence_calculator
    ):
        """Test no transition with low confidence."""
        factors = ConfidenceFactors(
            conversation_state=ConversationState.CONVERSING,
            llm_confidence=0.3,
            silence_duration=1.0,
            speech_clarity=0.9,
            person_distance=1.0,
            conversation_duration=10.0,
            turn_count=2,
            last_user_utterance_length=15,
        )

        result = confidence_calculator.calculate_completion_confidence(factors)
        should_transition = confidence_calculator.should_transition_to_concluding(
            result
        )

        # Low confidence should not trigger transition
        assert should_transition is False

    def test_should_transition_to_finished_very_high_confidence(
        self, confidence_calculator
    ):
        """Test immediate transition to finished with very high confidence."""
        factors = ConfidenceFactors(
            conversation_state=ConversationState.FINISHED,
            llm_confidence=0.95,
            silence_duration=8.0,
            speech_clarity=0.9,
            person_distance=5.0,
            conversation_duration=40.0,
            turn_count=5,
            last_user_utterance_length=2,
        )

        result = confidence_calculator.calculate_completion_confidence(factors)
        should_transition = confidence_calculator.should_transition_to_finished(
            result, time_in_concluding=1.0
        )

        # Very high confidence should trigger immediate transition
        assert should_transition is True

    def test_should_transition_to_finished_with_silence(self, confidence_calculator):
        """Test transition to finished after silence in concluding state."""
        factors = ConfidenceFactors(
            conversation_state=ConversationState.CONCLUDING,
            llm_confidence=0.7,
            silence_duration=10.0,  # High silence
            speech_clarity=0.9,
            person_distance=2.0,
            conversation_duration=35.0,
            turn_count=4,
            last_user_utterance_length=3,
        )

        result = confidence_calculator.calculate_completion_confidence(factors)
        should_transition = confidence_calculator.should_transition_to_finished(
            result, time_in_concluding=4.0  # Been concluding for a while
        )

        # High silence after being in concluding should trigger transition
        assert should_transition is True

    def test_should_transition_to_finished_timeout(self, confidence_calculator):
        """Test transition to finished after timeout."""
        factors = ConfidenceFactors(
            conversation_state=ConversationState.CONCLUDING,
            llm_confidence=0.65,
            silence_duration=3.0,
            speech_clarity=0.9,
            person_distance=2.0,
            conversation_duration=40.0,
            turn_count=5,
            last_user_utterance_length=5,
        )

        result = confidence_calculator.calculate_completion_confidence(factors)
        should_transition = confidence_calculator.should_transition_to_finished(
            result, time_in_concluding=6.0  # Timeout threshold
        )

        # Timeout with reasonable confidence should trigger transition
        assert should_transition is True

    def test_should_not_transition_to_finished_low_confidence(
        self, confidence_calculator
    ):
        """Test no transition to finished with low confidence and short time."""
        factors = ConfidenceFactors(
            conversation_state=ConversationState.CONCLUDING,
            llm_confidence=0.4,
            silence_duration=1.0,
            speech_clarity=0.9,
            person_distance=1.0,
            conversation_duration=15.0,
            turn_count=2,
            last_user_utterance_length=10,
        )

        result = confidence_calculator.calculate_completion_confidence(factors)
        should_transition = confidence_calculator.should_transition_to_finished(
            result, time_in_concluding=1.0
        )

        assert should_transition is False


class TestGreetingConversationStateMachineProvider:
    """Test suite for GreetingConversationStateMachineProvider."""

    def test_initialization(self, state_machine):
        """Test state machine initializes correctly."""
        assert state_machine.current_state == ConversationState.CONVERSING
        assert state_machine.previous_state is None
        assert state_machine.turn_count == 0
        assert state_machine.last_user_utterance == ""
        assert state_machine.confidence_history == []

    def test_start_conversation(self, state_machine):
        """Test starting a new conversation."""
        state_machine.start_conversation()

        assert state_machine.conversation_start_time is not None
        assert state_machine.turn_count == 0
        assert state_machine.confidence_history == []
        assert state_machine.current_state == ConversationState.CONVERSING

    def test_reset_state(self, state_machine):
        """Test resetting state machine."""
        state_machine.turn_count = 5
        state_machine.last_user_utterance = "test"
        state_machine.confidence_history = [0.5, 0.6, 0.7]

        state_machine.reset_state(ConversationState.IDLE)

        assert state_machine.current_state == ConversationState.CONVERSING
        assert state_machine.turn_count == 0
        assert state_machine.last_user_utterance == ""
        assert state_machine.confidence_history == []

    def test_process_conversation_basic(self, state_machine):
        """Test basic conversation processing."""
        state_machine.start_conversation()

        llm_output = {
            "conversation_state": ConversationState.CONVERSING,
            "confidence": 0.7,
            "speech_clarity": 0.9,
        }

        result = state_machine.process_conversation(llm_output)

        assert "current_state" in result
        assert "confidence" in result
        assert "command" in result
        assert "time_in_state" in result
        assert "confidence_trend" in result
        assert result["current_state"] == ConversationState.CONCLUDING.value

    def test_process_conversation_with_voice_input(self, state_machine_with_mock_io):
        """Test processing conversation with voice input."""
        state_machine, mock_io = state_machine_with_mock_io
        state_machine.start_conversation()

        voice_input = Input(input="Hello robot", timestamp=time.time(), tick=0)
        mock_io.get_input.return_value = voice_input
        mock_io.tick_counter = 0

        llm_output = {
            "conversation_state": ConversationState.CONVERSING,
            "confidence": 0.7,
            "speech_clarity": 0.9,
        }

        state_machine.process_conversation(llm_output)

        assert state_machine.turn_count == 1
        assert state_machine.last_user_utterance == "Hello robot"

    def test_state_transition_to_concluding(self, state_machine):
        """Test transition from conversing to concluding."""
        state_machine.start_conversation()
        state_machine.current_state = ConversationState.CONVERSING

        llm_output = {
            "conversation_state": ConversationState.CONCLUDING,
            "confidence": 0.9,  # High confidence
            "speech_clarity": 0.9,
        }

        result = state_machine.process_conversation(llm_output)

        # Should transition to concluding with high confidence
        assert result["current_state"] == ConversationState.CONCLUDING.value
        assert state_machine.previous_state == ConversationState.CONVERSING

    def test_state_transition_to_finished(self, state_machine):
        """Test transition from concluding to finished."""
        state_machine.start_conversation()
        # Set up a conversation with some history
        state_machine.turn_count = 3
        state_machine.conversation_start_time = (
            time.time() - 20.0
        )  # 20 seconds of conversation
        state_machine.current_state = ConversationState.CONCLUDING
        state_machine.state_entry_time = (
            time.time() - 6.0
        )  # Been concluding for 6 seconds

        llm_output = {
            "conversation_state": ConversationState.FINISHED,
            "confidence": 0.95,  # Very high confidence
            "speech_clarity": 0.9,
        }

        result = state_machine.process_conversation(llm_output)

        # Should transition to finished (timeout condition: 6s > 5s with overall >= 0.6)
        assert result["current_state"] == ConversationState.FINISHED.value

    def test_revert_to_conversing_from_concluding(self, state_machine):
        """Test reverting to conversing from concluding."""
        state_machine.start_conversation()
        state_machine.current_state = ConversationState.CONCLUDING

        llm_output = {
            "conversation_state": ConversationState.CONVERSING,  # LLM says still conversing
            "confidence": 0.5,
            "speech_clarity": 0.9,
        }

        result = state_machine.process_conversation(llm_output)

        # Should revert to conversing
        assert result["current_state"] == ConversationState.CONVERSING.value

    def test_emergency_timeout(self, state_machine):
        """Test emergency timeout forces finish."""
        state_machine.start_conversation()
        state_machine.current_state = ConversationState.CONCLUDING
        state_machine.state_entry_time = (
            time.time() - 20.0
        )  # Been concluding for 20 seconds

        llm_output = {
            "conversation_state": ConversationState.CONCLUDING,
            "confidence": 0.3,  # Even with low confidence
            "speech_clarity": 0.9,
        }

        result = state_machine.process_conversation(llm_output)

        # Emergency timeout should force finish
        assert result["current_state"] == ConversationState.FINISHED.value

    def test_confidence_trend_increasing(self, state_machine):
        """Test confidence trend detection - increasing."""
        state_machine.confidence_history = [0.4, 0.5, 0.7]

        trend = state_machine._get_confidence_trend()

        assert trend == "increasing"

    def test_confidence_trend_decreasing(self, state_machine):
        """Test confidence trend detection - decreasing."""
        state_machine.confidence_history = [0.8, 0.6, 0.4]

        trend = state_machine._get_confidence_trend()

        assert trend == "decreasing"

    def test_confidence_trend_stable(self, state_machine):
        """Test confidence trend detection - stable."""
        state_machine.confidence_history = [0.5, 0.52, 0.53]

        trend = state_machine._get_confidence_trend()

        assert trend == "stable"

    def test_confidence_trend_insufficient_data(self, state_machine):
        """Test confidence trend with insufficient data."""
        state_machine.confidence_history = [0.5]

        trend = state_machine._get_confidence_trend()

        assert trend == "insufficient_data"

    def test_generate_command_conversing(self, state_machine):
        """Test command generation in conversing state."""
        state_machine.current_state = ConversationState.CONVERSING

        command = state_machine._get_base_command()

        assert command["action"] == "stay_with_person"
        assert command["track_current"] is True
        assert command["find_new_target"] is False

    def test_generate_command_concluding(self, state_machine):
        """Test command generation in concluding state."""
        state_machine.current_state = ConversationState.CONCLUDING

        command = state_machine._get_base_command()

        assert command["action"] == "maintain_position"
        assert command["track_current"] is True
        assert command["find_new_target"] is False

    def test_generate_command_finished(self, state_machine):
        """Test command generation in finished state."""
        state_machine.current_state = ConversationState.FINISHED

        command = state_machine._get_base_command()

        assert command["action"] == "move_to_next_person"
        assert command["track_current"] is False
        assert command["find_new_target"] is True

    def test_generate_command_idle(self, state_machine):
        """Test command generation in idle state."""
        state_machine.current_state = ConversationState.IDLE

        command = state_machine._get_base_command()

        assert command["action"] == "wait"
        assert command["track_current"] is True
        assert command["find_new_target"] is False

    def test_get_conversation_duration(self, state_machine):
        """Test getting conversation duration."""
        # Clear conversation_start_time
        state_machine.conversation_start_time = None
        duration = state_machine._get_conversation_duration()
        assert duration == 0.0

        # Start conversation
        state_machine.start_conversation()
        time.sleep(0.1)  # Wait a bit

        duration = state_machine._get_conversation_duration()
        assert duration >= 0.1  # Should be at least the sleep time
        assert duration < 1.0  # But not too long

    def test_update_state_without_llm(self, state_machine_with_mock_io):
        """Test updating state without LLM input."""
        state_machine, mock_io = state_machine_with_mock_io
        state_machine.start_conversation()

        result = state_machine.update_state_without_llm()

        assert "current_state" in result
        assert "confidence" in result
        assert "command" in result
        assert "time_in_state" in result
        assert "confidence_trend" in result
        assert "silence_duration" in result

    def test_update_state_without_llm_with_voice(self, state_machine_with_mock_io):
        """Test updating state without LLM but with voice input."""
        state_machine, mock_io = state_machine_with_mock_io
        state_machine.start_conversation()

        # Mock voice input with old timestamp
        voice_timestamp = time.time() - 2.5
        voice_input = Input(input="test", timestamp=voice_timestamp, tick=0)
        mock_io.get_input.return_value = voice_input

        result = state_machine.update_state_without_llm()

        # Silence duration should be calculated from voice timestamp
        assert result["silence_duration"] >= 2.0
        assert result["silence_duration"] < 3.0

    def test_confidence_history_max_length(self, state_machine):
        """Test confidence history respects max length."""
        state_machine.start_conversation()

        for i in range(10):
            llm_output = {
                "conversation_state": ConversationState.CONVERSING,
                "confidence": 0.5 + i * 0.05,
                "speech_clarity": 0.9,
            }
            state_machine.process_conversation(llm_output)

        assert len(state_machine.confidence_history) == state_machine.max_history

    def test_command_includes_confidence_metadata(self, state_machine):
        """Test that generated commands include confidence metadata."""
        state_machine.start_conversation()

        llm_output = {
            "conversation_state": ConversationState.CONVERSING,
            "confidence": 0.7,
            "speech_clarity": 0.9,
        }

        result = state_machine.process_conversation(llm_output)

        assert "command" in result
        assert "confidence_metadata" in result["command"]
        assert "overall" in result["command"]["confidence_metadata"]
        assert "breakdown" in result["command"]["confidence_metadata"]
        assert "trend" in result["command"]["confidence_metadata"]

    def test_state_logging_on_transition(self, state_machine, caplog):
        """Test that state transitions are logged."""
        import logging

        caplog.set_level(logging.INFO)

        state_machine.start_conversation()
        state_machine.current_state = ConversationState.CONVERSING

        llm_output = {
            "conversation_state": ConversationState.CONCLUDING,
            "confidence": 0.9,
            "speech_clarity": 0.9,
        }

        state_machine.process_conversation(llm_output)

        assert any("State transition" in record.message for record in caplog.records)

    def test_multiple_state_transitions(self, state_machine):
        """Test multiple state transitions in sequence."""
        state_machine.start_conversation()

        # Start in conversing
        assert state_machine.current_state == ConversationState.CONVERSING

        # Transition to concluding
        llm_output = {
            "conversation_state": ConversationState.CONCLUDING,
            "confidence": 0.9,
            "speech_clarity": 0.9,
        }
        state_machine.process_conversation(llm_output)
        assert state_machine.current_state == ConversationState.CONCLUDING

        # Wait and transition to finished with very high confidence
        state_machine.state_entry_time = time.time() - 10.0
        # Ensure conversation has enough history for quality score
        state_machine.turn_count = 3
        state_machine.conversation_start_time = time.time() - 25.0
        llm_output = {
            "conversation_state": ConversationState.FINISHED,
            "confidence": 0.95,  # Very high confidence
            "speech_clarity": 0.9,
        }
        result = state_machine.process_conversation(llm_output)
        assert result["current_state"] == ConversationState.FINISHED.value


class TestGreetingConversationStateIntegration:
    """Integration tests for the greeting conversation state system."""

    def test_full_conversation_flow(self, state_machine_with_mock_io):
        """Test a complete conversation flow from start to finish."""
        state_machine, mock_io = state_machine_with_mock_io

        # Start conversation
        state_machine.start_conversation()
        assert state_machine.current_state == ConversationState.CONVERSING

        # Simulate several conversation turns
        for i in range(3):
            voice_input = Input(
                input=f"User message {i}", timestamp=time.time(), tick=i
            )
            mock_io.get_input.return_value = voice_input
            mock_io.tick_counter = i

            llm_output = {
                "conversation_state": ConversationState.CONVERSING,
                "confidence": 0.6,
                "speech_clarity": 0.9,
            }
            state_machine.process_conversation(llm_output)
            time.sleep(0.01)

        # Conversation should still be active
        assert state_machine.current_state == ConversationState.CONVERSING
        assert state_machine.turn_count == 3

        # Now signal to conclude
        llm_output = {
            "conversation_state": ConversationState.CONCLUDING,
            "confidence": 0.9,
            "speech_clarity": 0.9,
        }
        result = state_machine.process_conversation(llm_output)
        assert result["current_state"] == ConversationState.CONCLUDING.value

        # After some silence, should finish with very high confidence
        state_machine.state_entry_time = time.time() - 6.0
        # Ensure conversation has enough history for quality score
        state_machine.conversation_start_time = time.time() - 30.0
        llm_output = {
            "conversation_state": ConversationState.FINISHED,
            "confidence": 0.95,  # Very high confidence
            "speech_clarity": 0.9,
        }
        result = state_machine.process_conversation(llm_output)
        assert result["current_state"] == ConversationState.FINISHED.value

    def test_conversation_with_mixed_signals(self, state_machine):
        """Test conversation handling with mixed confidence signals."""
        state_machine.start_conversation()

        # Low LLM confidence but other factors suggest continuation
        llm_output = {
            "conversation_state": ConversationState.CONVERSING,
            "confidence": 0.3,  # Low confidence
            "speech_clarity": 0.95,  # But good clarity
        }
        result = state_machine.process_conversation(llm_output)

        assert result["current_state"] == ConversationState.CONVERSING.value

    def test_rapid_state_changes(self, state_machine):
        """Test handling of rapid state changes."""
        state_machine.start_conversation()

        states = [
            ConversationState.CONVERSING,
            ConversationState.CONCLUDING,
            ConversationState.CONVERSING,  # Revert
            ConversationState.CONCLUDING,
        ]

        result = None
        for conv_state in states:
            llm_output = {
                "conversation_state": conv_state,
                "confidence": 0.8,
                "speech_clarity": 0.9,
            }
            result = state_machine.process_conversation(llm_output)
            time.sleep(0.01)

        assert result is not None
        assert result["current_state"] in [
            ConversationState.CONVERSING.value,
            ConversationState.CONCLUDING.value,
        ]
