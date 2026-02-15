from actions.move_go2_teleops.interface import Move, MoveInput, MovementAction


class TestMovementAction:
    """Tests for the MovementAction enum."""

    def test_movement_action_values(self):
        """Test that MovementAction enum has correct values."""
        assert MovementAction.STAND_UP.value == "stand up"
        assert MovementAction.SIT.value == "sit"
        assert MovementAction.SHAKE_PAW.value == "shake paw"
        assert MovementAction.DANCE.value == "dance"
        assert MovementAction.STRETCH.value == "stretch"
        assert MovementAction.STAND_STILL.value == "stand still"
        assert MovementAction.DO_NOTHING.value == "stand still"

    def test_movement_action_is_string_enum(self):
        """Test that MovementAction values are strings."""
        for action in MovementAction:
            assert isinstance(action.value, str)

    def test_movement_action_count(self):
        """Test that MovementAction has expected number of actions."""
        assert len(MovementAction) == 6


class TestMoveInput:
    """Tests for the MoveInput dataclass."""

    def test_move_input_creation(self):
        """Test creating MoveInput with valid action."""
        move_input = MoveInput(action=MovementAction.STAND_UP)
        assert move_input.action == MovementAction.STAND_UP

    def test_move_input_all_actions(self):
        """Test creating MoveInput with all possible actions."""
        for action in MovementAction:
            move_input = MoveInput(action=action)
            assert move_input.action == action


class TestMove:
    """Tests for the Move interface."""

    def test_move_creation(self):
        """Test creating Move with input and output."""
        move_input = MoveInput(action=MovementAction.DANCE)
        move = Move(input=move_input, output=move_input)
        assert move.input == move_input
        assert move.output == move_input

    def test_move_different_input_output(self):
        """Test creating Move with different input and output."""
        input_move = MoveInput(action=MovementAction.STAND_UP)
        output_move = MoveInput(action=MovementAction.SIT)
        move = Move(input=input_move, output=output_move)
        assert move.input.action == MovementAction.STAND_UP
        assert move.output.action == MovementAction.SIT
