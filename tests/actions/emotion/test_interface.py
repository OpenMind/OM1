"""Tests for the Emotion action interface."""

from actions.emotion.interface import Emotion, EmotionAction, EmotionInput


class TestEmotionAction:
    """Tests for the EmotionAction enum."""

    def test_emotion_action_values(self):
        """Test that EmotionAction has expected values."""
        assert EmotionAction.HAPPY == "happy"
        assert EmotionAction.SAD == "sad"
        assert EmotionAction.MAD == "mad"
        assert EmotionAction.CURIOUS == "curious"

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
        """Test creating EmotionInput with action."""
        emotion_input = EmotionInput(action=EmotionAction.HAPPY)
        assert emotion_input.action == EmotionAction.HAPPY

    def test_emotion_input_all_actions(self):
        """Test creating EmotionInput with all emotion types."""
        for action in EmotionAction:
            emotion_input = EmotionInput(action=action)
            assert emotion_input.action == action


class TestEmotion:
    """Tests for the Emotion interface."""

    def test_emotion_creation(self):
        """Test creating Emotion with input and output."""
        emotion_input = EmotionInput(action=EmotionAction.CURIOUS)
        emotion = Emotion(input=emotion_input, output=emotion_input)
        assert emotion.input == emotion_input
        assert emotion.output == emotion_input

    def test_emotion_different_input_output(self):
        """Test creating Emotion with different input and output."""
        input_emotion = EmotionInput(action=EmotionAction.SAD)
        output_emotion = EmotionInput(action=EmotionAction.HAPPY)
        emotion = Emotion(input=input_emotion, output=output_emotion)
        assert emotion.input.action == EmotionAction.SAD
        assert emotion.output.action == EmotionAction.HAPPY
