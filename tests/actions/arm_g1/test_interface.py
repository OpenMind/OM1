"""Tests for the Arm G1 action interface."""

from actions.arm_g1.interface import Arm, ArmAction, ArmInput


class TestArmAction:
    """Tests for the ArmAction enum."""

    def test_arm_action_values(self):
        """Test that ArmAction has expected values."""
        assert ArmAction.IDLE == "idle"
        assert ArmAction.LEFT_KISS == "left kiss"
        assert ArmAction.RIGHT_KISS == "right kiss"
        assert ArmAction.CLAP == "clap"
        assert ArmAction.HIGH_FIVE == "high five"
        assert ArmAction.SHAKE_HAND == "shake hand"
        assert ArmAction.HEART == "heart"
        assert ArmAction.HIGH_WAVE == "high wave"

    def test_arm_action_is_string_enum(self):
        """Test that ArmAction values are strings."""
        for action in ArmAction:
            assert isinstance(action.value, str)

    def test_arm_action_count(self):
        """Test that ArmAction has expected number of actions."""
        assert len(ArmAction) == 8


class TestArmInput:
    """Tests for the ArmInput dataclass."""

    def test_arm_input_creation(self):
        """Test creating ArmInput with action."""
        arm_input = ArmInput(action=ArmAction.CLAP)
        assert arm_input.action == ArmAction.CLAP

    def test_arm_input_all_actions(self):
        """Test creating ArmInput with all action types."""
        for action in ArmAction:
            arm_input = ArmInput(action=action)
            assert arm_input.action == action


class TestArm:
    """Tests for the Arm interface."""

    def test_arm_creation(self):
        """Test creating Arm with input and output."""
        arm_input = ArmInput(action=ArmAction.HIGH_FIVE)
        arm = Arm(input=arm_input, output=arm_input)
        assert arm.input == arm_input
        assert arm.output == arm_input

    def test_arm_different_input_output(self):
        """Test creating Arm with different input and output."""
        input_action = ArmInput(action=ArmAction.HIGH_WAVE)
        output_action = ArmInput(action=ArmAction.IDLE)
        arm = Arm(input=input_action, output=output_action)
        assert arm.input.action == ArmAction.HIGH_WAVE
        assert arm.output.action == ArmAction.IDLE
