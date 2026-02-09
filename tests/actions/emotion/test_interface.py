from actions.emotion.interface import Emotion, EmotionAction, EmotionInput


class TestEmotionAction:
    """Tests for the EmotionAction enum."""

    def test_emotion_action_values(self):
        """Test that EmotionAction enum has correct values."""
        assert EmotionAction.HAPPY.value == "happy"
        assert EmotionAction.SAD.value == "sad"
        assert EmotionAction.MAD.value == "mad"
        assert EmotionAction.CURIOUS.value == "curious"

    def test_emotion_action_is_string_enum(self):
        """Test that EmotionAction values are strings."""
        for action in EmotionAction:
            assert isinstance(action.value, str)

    def test_emotion_action_count(self):
        """Test that EmotionAction has expected number of emotions."""
        assert len(EmotionAction) == 4


class TestEmotionInput:
    """Tests for the EmotionInput dataclass."""

    def test_emotion_input_creation(self):
        """Test creating EmotionInput with valid action."""
        emotion_input = EmotionInput(action=EmotionAction.HAPPY)
        assert emotion_input.action == EmotionAction.HAPPY

    def test_emotion_input_all_actions(self):
        """Test creating EmotionInput with all possible actions."""
        for action in EmotionAction:
            emotion_input = EmotionInput(action=action)
            assert emotion_input.action == action


class TestEmotion:
    """Tests for the Emotion interface."""

    def test_emotion_creation(self):
        """Test creating Emotion with input and output."""
        emotion_input = EmotionInput(action=EmotionAction.HAPPY)
        emotion = Emotion(input=emotion_input, output=emotion_input)
        assert emotion.input == emotion_input
        assert emotion.output == emotion_input

    def test_emotion_different_input_output(self):
        """Test creating Emotion with different input and output."""
        input_emotion = EmotionInput(action=EmotionAction.HAPPY)
        output_emotion = EmotionInput(action=EmotionAction.SAD)
        emotion = Emotion(input=input_emotion, output=output_emotion)
        assert emotion.input.action == EmotionAction.HAPPY
        assert emotion.output.action == EmotionAction.SAD
