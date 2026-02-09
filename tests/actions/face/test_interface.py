from actions.face.interface import Face, FaceAction, FaceInput


class TestFaceAction:
    """Tests for the FaceAction enum."""

    def test_face_action_values(self):
        """Test that FaceAction enum has correct values."""
        assert FaceAction.HAPPY.value == "happy"
        assert FaceAction.CONFUSED.value == "confused"
        assert FaceAction.CURIOUS.value == "curious"
        assert FaceAction.EXCITED.value == "excited"
        assert FaceAction.SAD.value == "sad"
        assert FaceAction.THINK.value == "think"

    def test_face_action_is_string_enum(self):
        """Test that FaceAction values are strings."""
        for action in FaceAction:
            assert isinstance(action.value, str)

    def test_face_action_count(self):
        """Test that FaceAction has expected number of expressions."""
        assert len(FaceAction) == 6


class TestFaceInput:
    """Tests for the FaceInput dataclass."""

    def test_face_input_creation(self):
        """Test creating FaceInput with valid action."""
        face_input = FaceInput(action=FaceAction.HAPPY)
        assert face_input.action == FaceAction.HAPPY

    def test_face_input_all_actions(self):
        """Test creating FaceInput with all possible actions."""
        for action in FaceAction:
            face_input = FaceInput(action=action)
            assert face_input.action == action


class TestFace:
    """Tests for the Face interface."""

    def test_face_creation(self):
        """Test creating Face with input and output."""
        face_input = FaceInput(action=FaceAction.HAPPY)
        face = Face(input=face_input, output=face_input)
        assert face.input == face_input
        assert face.output == face_input

    def test_face_different_input_output(self):
        """Test creating Face with different input and output."""
        input_face = FaceInput(action=FaceAction.HAPPY)
        output_face = FaceInput(action=FaceAction.SAD)
        face = Face(input=input_face, output=output_face)
        assert face.input.action == FaceAction.HAPPY
        assert face.output.action == FaceAction.SAD
