from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class EmotionAction(str, Enum):
    """
    Enumeration of possible emotional states for agent expression.

    This enum defines the set of emotions that an agent can express through
    physical displays (e.g., LED patterns, screen animations) or behavioral
    changes. Each emotion maps to a string value used by the underlying
    hardware or simulation systems.

    Attributes
    ----------
    HAPPY : str
        Expresses joy, satisfaction, or positive engagement. Typically displayed
        through upbeat animations, bright colors, or cheerful sounds.
    SAD : str
        Expresses disappointment, empathy, or low energy states. Often shown
        through subdued colors, slower animations, or downcast displays.
    MAD : str
        Expresses frustration, displeasure, or strong disagreement. Usually
        indicated through intense colors, sharp animations, or alert sounds.
    CURIOUS : str
        Expresses interest, inquisitiveness, or engagement with something new.
        Typically shown through attentive animations or searching behaviors.

    Examples
    --------
    >>> from actions.emotion.interface import EmotionAction
    >>> emotion = EmotionAction.HAPPY
    >>> print(emotion.value)
    'happy'

    >>> # Check if an emotion is valid
    >>> EmotionAction("curious") == EmotionAction.CURIOUS
    True
    """

    HAPPY = "happy"
    SAD = "sad"
    MAD = "mad"
    CURIOUS = "curious"


@dataclass
class EmotionInput:
    """
    Input interface for the Emotion action.

    This dataclass defines the input structure required to trigger an emotional
    expression on the agent. The emotion is selected from the predefined set
    of EmotionAction values.

    Parameters
    ----------
    action : EmotionAction
        The emotional state to express. Must be one of the valid EmotionAction
        enum values: HAPPY, SAD, MAD, or CURIOUS.

    Examples
    --------
    >>> from actions.emotion.interface import EmotionInput, EmotionAction
    >>> # Create an input to express happiness
    >>> happy_input = EmotionInput(action=EmotionAction.HAPPY)

    >>> # Create an input to express curiosity
    >>> curious_input = EmotionInput(action=EmotionAction.CURIOUS)
    """

    action: EmotionAction


@dataclass
class Emotion(Interface[EmotionInput, EmotionInput]):
    """
    Action interface for expressing emotional states.

    This action allows an agent to display emotions through physical or visual
    expressions. The emotion system enables more natural human-robot interaction
    by providing non-verbal communication cues that complement speech and movement.

    Effect: Triggers an emotional expression on the agent's display system,
    which may include LED patterns, screen animations, sound effects, or
    physical movements depending on the robot's hardware capabilities.

    The emotion display typically persists until a new emotion is set or the
    agent returns to a neutral state. Emotions can be used to:
    - Respond to user interactions (e.g., showing happiness when greeted)
    - Indicate internal states (e.g., curiosity when exploring)
    - Provide feedback (e.g., sadness when unable to help)
    - Express reactions to environmental stimuli

    Parameters
    ----------
    input : EmotionInput
        The input containing the desired emotional state to express.
    output : EmotionInput
        The output echoing the emotion that was expressed. Matches the input
        to confirm the action was processed.

    Examples
    --------
    >>> from actions.emotion.interface import Emotion, EmotionInput, EmotionAction
    >>> # Create an emotion action to express happiness
    >>> emotion_input = EmotionInput(action=EmotionAction.HAPPY)
    >>> emotion_action = Emotion(input=emotion_input, output=emotion_input)

    See Also
    --------
    actions.face.interface.Face : Action for displaying facial expressions.
    actions.speak.interface.Speak : Action for verbal communication.
    """

    input: EmotionInput
    output: EmotionInput
