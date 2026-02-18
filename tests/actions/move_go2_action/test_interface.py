"""Tests for the Move Go2 Action interface."""

from actions.move_go2_action.interface import Action, ActionInput, Move


class TestAction:
    """Tests for the Action enum."""

    def test_action_values(self):
        """Test that Action has expected values."""
        assert Action.SHAKE_PAW == "shake paw"
        assert Action.DANCE == "dance"
        assert Action.STRETCH == "stretch"
        assert Action.STAND_STILL == "stand still"
        assert Action.DO_NOTHING == "stand still"

    def test_action_is_string_enum(self):
        """Test that Action values are strings."""
        for action in Action:
            assert isinstance(action.value, str)

    def test_action_count(self):
        """Test that Action has expected number of actions."""
        # DO_NOTHING is alias for STAND_STILL, so only 4 unique
        assert len(Action) == 4


class TestActionInput:
    """Tests for the ActionInput dataclass."""

    def test_action_input_creation(self):
        """Test creating ActionInput with action."""
        action_input = ActionInput(action=Action.DANCE)
        assert action_input.action == Action.DANCE

    def test_action_input_all_actions(self):
        """Test creating ActionInput with all action types."""
        for action in Action:
            action_input = ActionInput(action=action)
            assert action_input.action == action


class TestMove:
    """Tests for the Move interface."""

    def test_move_creation(self):
        """Test creating Move with input and output."""
        action_input = ActionInput(action=Action.STRETCH)
        move = Move(input=action_input, output=action_input)
        assert move.input == action_input
        assert move.output == action_input

    def test_move_different_input_output(self):
        """Test creating Move with different input and output."""
        input_action = ActionInput(action=Action.SHAKE_PAW)
        output_action = ActionInput(action=Action.STAND_STILL)
        move = Move(input=input_action, output=output_action)
        assert move.input.action == Action.SHAKE_PAW
        assert move.output.action == Action.STAND_STILL
