from actions.move_k1_autonomy.interface import Move, MoveInput, MovementAction


class TestMovementAction:
    """Test MovementAction enum."""

    def test_movement_action_values(self):
        """Test that all movement actions have correct string values."""
        assert MovementAction.TURN_LEFT == "turn left"
        assert MovementAction.TURN_RIGHT == "turn right"
        assert MovementAction.MOVE_FORWARDS == "move forwards"
        assert MovementAction.MOVE_BACK == "move back"
        assert MovementAction.STAND_STILL == "stand still"
        assert MovementAction.DO_NOTHING == "stand still"

    def test_movement_action_enum_members(self):
        """Test that all expected enum members exist."""
        actions = [action.value for action in MovementAction]
        assert "turn left" in actions
        assert "turn right" in actions
        assert "move forwards" in actions
        assert "move back" in actions
        assert "stand still" in actions

    def test_movement_action_is_string_enum(self):
        """Test that MovementAction values are strings."""
        for action in MovementAction:
            assert isinstance(action.value, str)

    def test_movement_action_aliases(self):
        """Test that DO_NOTHING is an alias for STAND_STILL."""
        assert MovementAction.DO_NOTHING == MovementAction.STAND_STILL
        assert MovementAction.DO_NOTHING.value == MovementAction.STAND_STILL.value


class TestMoveInput:
    """Test MoveInput dataclass."""

    def test_move_input_creation_turn_left(self):
        """Test creating MoveInput with turn left action."""
        move_input = MoveInput(action=MovementAction.TURN_LEFT)
        assert move_input.action == MovementAction.TURN_LEFT
        assert move_input.action == "turn left"

    def test_move_input_creation_turn_right(self):
        """Test creating MoveInput with turn right action."""
        move_input = MoveInput(action=MovementAction.TURN_RIGHT)
        assert move_input.action == MovementAction.TURN_RIGHT
        assert move_input.action == "turn right"

    def test_move_input_creation_move_forwards(self):
        """Test creating MoveInput with move forwards action."""
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
        assert move_input.action == MovementAction.MOVE_FORWARDS
        assert move_input.action == "move forwards"

    def test_move_input_creation_move_back(self):
        """Test creating MoveInput with move back action."""
        move_input = MoveInput(action=MovementAction.MOVE_BACK)
        assert move_input.action == MovementAction.MOVE_BACK
        assert move_input.action == "move back"

    def test_move_input_creation_stand_still(self):
        """Test creating MoveInput with stand still action."""
        move_input = MoveInput(action=MovementAction.STAND_STILL)
        assert move_input.action == MovementAction.STAND_STILL
        assert move_input.action == "stand still"

    def test_move_input_creation_do_nothing(self):
        """Test creating MoveInput with do nothing action."""
        move_input = MoveInput(action=MovementAction.DO_NOTHING)
        assert move_input.action == MovementAction.DO_NOTHING
        assert move_input.action == "stand still"

    def test_move_input_with_string_value(self):
        """Test creating MoveInput with string value directly."""
        move_input = MoveInput(action=MovementAction.TURN_LEFT)
        assert move_input.action == "turn left"

    def test_move_input_equality(self):
        """Test MoveInput equality comparison."""
        input1 = MoveInput(action=MovementAction.TURN_LEFT)
        input2 = MoveInput(action=MovementAction.TURN_LEFT)
        input3 = MoveInput(action=MovementAction.TURN_RIGHT)

        assert input1 == input2
        assert input1 != input3

    def test_move_input_is_dataclass(self):
        """Test that MoveInput is a dataclass."""
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
        assert hasattr(move_input, "__dataclass_fields__")


class TestMove:
    """Test Move interface."""

    def test_move_creation_same_input_output(self):
        """Test creating Move with same input and output."""
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
        move = Move(input=move_input, output=move_input)

        assert move.input == move_input
        assert move.output == move_input
        assert move.input.action == MovementAction.MOVE_FORWARDS
        assert move.output.action == MovementAction.MOVE_FORWARDS

    def test_move_creation_different_input_output(self):
        """Test creating Move with different input and output."""
        input_cmd = MoveInput(action=MovementAction.TURN_LEFT)
        output_cmd = MoveInput(action=MovementAction.MOVE_FORWARDS)
        move = Move(input=input_cmd, output=output_cmd)

        assert move.input == input_cmd
        assert move.output == output_cmd
        assert move.input.action == MovementAction.TURN_LEFT
        assert move.output.action == MovementAction.MOVE_FORWARDS

    def test_move_with_all_actions(self):
        """Test creating Move with all possible actions."""
        actions = [
            MovementAction.TURN_LEFT,
            MovementAction.TURN_RIGHT,
            MovementAction.MOVE_FORWARDS,
            MovementAction.MOVE_BACK,
            MovementAction.STAND_STILL,
        ]

        for action in actions:
            move_input = MoveInput(action=action)
            move = Move(input=move_input, output=move_input)
            assert move.input.action == action
            assert move.output.action == action

    def test_move_equality(self):
        """Test Move equality comparison."""
        input1 = MoveInput(action=MovementAction.TURN_LEFT)
        input2 = MoveInput(action=MovementAction.TURN_LEFT)
        input3 = MoveInput(action=MovementAction.TURN_RIGHT)

        move1 = Move(input=input1, output=input1)
        move2 = Move(input=input2, output=input2)
        move3 = Move(input=input3, output=input3)

        assert move1 == move2
        assert move1 != move3

    def test_move_is_dataclass(self):
        """Test that Move is a dataclass."""
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
        move = Move(input=move_input, output=move_input)
        assert hasattr(move, "__dataclass_fields__")

    def test_move_has_docstring(self):
        """Test that Move has a descriptive docstring."""
        assert Move.__doc__ is not None
        assert len(Move.__doc__.strip()) > 0
        # Check for key phrases in docstring
        docstring_lower = Move.__doc__.lower()
        assert "move" in docstring_lower or "action" in docstring_lower


class TestMovementActionIntegration:
    """Integration tests for movement actions."""

    def test_all_actions_can_create_move_input(self):
        """Test that all MovementAction values can create valid MoveInput."""
        for action in MovementAction:
            move_input = MoveInput(action=action)
            assert move_input.action == action.value

    def test_all_actions_can_create_move_interface(self):
        """Test that all MovementAction values can create valid Move interface."""
        for action in MovementAction:
            move_input = MoveInput(action=action)
            move = Move(input=move_input, output=move_input)
            assert move.input.action == action.value
            assert move.output.action == action.value

    def test_string_action_values_match_enum(self):
        """Test that string action values correctly match enum values."""
        string_actions = {
            "turn left": MovementAction.TURN_LEFT,
            "turn right": MovementAction.TURN_RIGHT,
            "move forwards": MovementAction.MOVE_FORWARDS,
            "move back": MovementAction.MOVE_BACK,
            "stand still": MovementAction.STAND_STILL,
        }

        for string_val, enum_val in string_actions.items():
            assert enum_val.value == string_val
            move_input = MoveInput(action=string_val)  # type: ignore
            assert move_input.action == string_val
