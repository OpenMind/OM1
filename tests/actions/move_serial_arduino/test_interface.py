"""Tests for the Move Serial Arduino interface."""

from actions.move_serial_arduino.interface import Move, MoveInput, MovementAction


class TestMovementAction:
    """Tests for the MovementAction enum."""

    def test_movement_action_values(self):
        """Test that MovementAction has expected values."""
        assert MovementAction.BE_STILL == "be still"
        assert MovementAction.JUMP_SMALL == "small jump"
        assert MovementAction.JUMP_MEDIUM == "medium jump"
        assert MovementAction.JUMP_BIG == "big jump"

    def test_movement_action_is_string_enum(self):
        """Test that MovementAction values are strings."""
        for action in MovementAction:
            assert isinstance(action.value, str)

    def test_movement_action_count(self):
        """Test that MovementAction has expected number of actions."""
        assert len(MovementAction) == 4


class TestMoveInput:
    """Tests for the MoveInput dataclass."""

    def test_move_input_creation(self):
        """Test creating MoveInput with action."""
        move_input = MoveInput(action=MovementAction.JUMP_SMALL)
        assert move_input.action == MovementAction.JUMP_SMALL

    def test_move_input_all_actions(self):
        """Test creating MoveInput with all action types."""
        for action in MovementAction:
            move_input = MoveInput(action=action)
            assert move_input.action == action


class TestMove:
    """Tests for the Move interface."""

    def test_move_creation(self):
        """Test creating Move with input and output."""
        move_input = MoveInput(action=MovementAction.JUMP_BIG)
        move = Move(input=move_input, output=move_input)
        assert move.input == move_input
        assert move.output == move_input

    def test_move_different_input_output(self):
        """Test creating Move with different input and output."""
        input_move = MoveInput(action=MovementAction.JUMP_MEDIUM)
        output_move = MoveInput(action=MovementAction.BE_STILL)
        move = Move(input=input_move, output=output_move)
        assert move.input.action == MovementAction.JUMP_MEDIUM
        assert move.output.action == MovementAction.BE_STILL
