"""Tests for the Move Go2 Autonomy interface."""

from actions.move_go2_autonomy.interface import Move, MoveInput, MovementAction


class TestMovementAction:
    """Tests for the MovementAction enum."""

    def test_movement_action_values(self):
        """Test that MovementAction has expected values."""
        assert MovementAction.TURN_LEFT == "turn left"
        assert MovementAction.TURN_RIGHT == "turn right"
        assert MovementAction.MOVE_FORWARDS == "move forwards"
        assert MovementAction.MOVE_BACK == "move back"
        assert MovementAction.STAND_STILL == "stand still"
        assert MovementAction.DO_NOTHING == "stand still"

    def test_movement_action_is_string_enum(self):
        """Test that MovementAction values are strings."""
        for action in MovementAction:
            assert isinstance(action.value, str)

    def test_movement_action_count(self):
        """Test that MovementAction has expected number of actions."""
        # DO_NOTHING is alias for STAND_STILL, so only 5 unique
        assert len(MovementAction) == 5


class TestMoveInput:
    """Tests for the MoveInput dataclass."""

    def test_move_input_creation(self):
        """Test creating MoveInput with action."""
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
        assert move_input.action == MovementAction.MOVE_FORWARDS

    def test_move_input_all_actions(self):
        """Test creating MoveInput with all action types."""
        for action in MovementAction:
            move_input = MoveInput(action=action)
            assert move_input.action == action


class TestMove:
    """Tests for the Move interface."""

    def test_move_creation(self):
        """Test creating Move with input and output."""
        move_input = MoveInput(action=MovementAction.TURN_LEFT)
        move = Move(input=move_input, output=move_input)
        assert move.input == move_input
        assert move.output == move_input

    def test_move_different_input_output(self):
        """Test creating Move with different input and output."""
        input_move = MoveInput(action=MovementAction.MOVE_FORWARDS)
        output_move = MoveInput(action=MovementAction.STAND_STILL)
        move = Move(input=input_move, output=output_move)
        assert move.input.action == MovementAction.MOVE_FORWARDS
        assert move.output.action == MovementAction.STAND_STILL
