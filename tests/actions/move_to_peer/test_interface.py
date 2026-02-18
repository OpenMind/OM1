"""Tests for the Move To Peer interface."""

from actions.move_to_peer.interface import MoveToPeer, MoveToPeerAction, MoveToPeerInput


class TestMoveToPeerAction:
    """Tests for the MoveToPeerAction enum."""

    def test_move_to_peer_action_values(self):
        """Test that MoveToPeerAction has expected values."""
        assert MoveToPeerAction.IDLE == "idle"
        assert MoveToPeerAction.NAVIGATE == "navigate"

    def test_move_to_peer_action_is_string_enum(self):
        """Test that MoveToPeerAction values are strings."""
        for action in MoveToPeerAction:
            assert isinstance(action.value, str)

    def test_move_to_peer_action_count(self):
        """Test that MoveToPeerAction has expected number of actions."""
        assert len(MoveToPeerAction) == 2


class TestMoveToPeerInput:
    """Tests for the MoveToPeerInput dataclass."""

    def test_move_to_peer_input_creation(self):
        """Test creating MoveToPeerInput with action."""
        move_input = MoveToPeerInput(action=MoveToPeerAction.NAVIGATE)
        assert move_input.action == MoveToPeerAction.NAVIGATE

    def test_move_to_peer_input_all_actions(self):
        """Test creating MoveToPeerInput with all action types."""
        for action in MoveToPeerAction:
            move_input = MoveToPeerInput(action=action)
            assert move_input.action == action


class TestMoveToPeer:
    """Tests for the MoveToPeer interface."""

    def test_move_to_peer_creation(self):
        """Test creating MoveToPeer with input and output."""
        move_input = MoveToPeerInput(action=MoveToPeerAction.NAVIGATE)
        move = MoveToPeer(input=move_input, output=move_input)
        assert move.input == move_input
        assert move.output == move_input

    def test_move_to_peer_different_input_output(self):
        """Test creating MoveToPeer with different input and output."""
        input_move = MoveToPeerInput(action=MoveToPeerAction.NAVIGATE)
        output_move = MoveToPeerInput(action=MoveToPeerAction.IDLE)
        move = MoveToPeer(input=input_move, output=output_move)
        assert move.input.action == MoveToPeerAction.NAVIGATE
        assert move.output.action == MoveToPeerAction.IDLE
