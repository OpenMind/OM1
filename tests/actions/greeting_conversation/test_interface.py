"""Tests for the GreetingConversation action interface."""

from actions.greeting_conversation.interface import (
    ConversationState,
    GreetingConversation,
    GreetingConversationInput,
)


class TestConversationState:
    """Tests for the ConversationState enum."""

    def test_conversation_state_values(self):
        """Test that ConversationState has expected values."""
        assert ConversationState.CONVERSING.value == "conversing"
        assert ConversationState.CONCLUDING.value == "concluding"
        assert ConversationState.FINISHED.value == "finished"

    def test_conversation_state_count(self):
        """Test that ConversationState has expected number of states."""
        assert len(ConversationState) == 3


class TestGreetingConversationInput:
    """Tests for the GreetingConversationInput dataclass."""

    def test_greeting_conversation_input_creation(self):
        """Test creating GreetingConversationInput."""
        gc_input = GreetingConversationInput(
            response="Hello! How can I help you?",
            conversation_state=ConversationState.CONVERSING,
            confidence=0.95,
            speech_clarity=0.85,
        )
        assert gc_input.response == "Hello! How can I help you?"
        assert gc_input.conversation_state == ConversationState.CONVERSING
        assert gc_input.confidence == 0.95
        assert gc_input.speech_clarity == 0.85

    def test_greeting_conversation_input_all_states(self):
        """Test creating GreetingConversationInput with all states."""
        for state in ConversationState:
            gc_input = GreetingConversationInput(
                response="Test",
                conversation_state=state,
                confidence=0.5,
                speech_clarity=0.5,
            )
            assert gc_input.conversation_state == state


class TestGreetingConversation:
    """Tests for the GreetingConversation interface."""

    def test_greeting_conversation_creation(self):
        """Test creating GreetingConversation with input and output."""
        gc_input = GreetingConversationInput(
            response="Goodbye!",
            conversation_state=ConversationState.FINISHED,
            confidence=0.9,
            speech_clarity=0.8,
        )
        gc = GreetingConversation(input=gc_input, output=gc_input)
        assert gc.input == gc_input
        assert gc.output == gc_input

    def test_greeting_conversation_different_input_output(self):
        """Test creating GreetingConversation with different input and output."""
        input_gc = GreetingConversationInput(
            response="Hello",
            conversation_state=ConversationState.CONVERSING,
            confidence=0.7,
            speech_clarity=0.6,
        )
        output_gc = GreetingConversationInput(
            response="Goodbye",
            conversation_state=ConversationState.FINISHED,
            confidence=0.9,
            speech_clarity=0.8,
        )
        gc = GreetingConversation(input=input_gc, output=output_gc)
        assert gc.input.conversation_state == ConversationState.CONVERSING
        assert gc.output.conversation_state == ConversationState.FINISHED
