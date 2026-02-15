from actions.move_game_controller.interface import GameController, IDLEInput


class TestIDLEInput:
    """Tests for the IDLEInput dataclass."""

    def test_idle_input_creation(self):
        """Test creating IDLEInput with action."""
        idle_input = IDLEInput(action="test_command")
        assert idle_input.action == "test_command"

    def test_idle_input_empty_action(self):
        """Test creating IDLEInput with empty action."""
        idle_input = IDLEInput(action="")
        assert idle_input.action == ""


class TestGameController:
    """Tests for the GameController interface."""

    def test_game_controller_creation(self):
        """Test creating GameController with input and output."""
        idle_input = IDLEInput(action="move_forward")
        controller = GameController(input=idle_input, output=idle_input)
        assert controller.input == idle_input
        assert controller.output == idle_input

    def test_game_controller_different_input_output(self):
        """Test creating GameController with different input and output."""
        input_cmd = IDLEInput(action="input_command")
        output_cmd = IDLEInput(action="output_command")
        controller = GameController(input=input_cmd, output=output_cmd)
        assert controller.input.action == "input_command"
        assert controller.output.action == "output_command"
