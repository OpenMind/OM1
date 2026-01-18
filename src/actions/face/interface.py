from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class FaceAction(str, Enum):
    """
    Enumeration of possible facial expressions for visual display.

    This enum defines the set of facial expressions that an agent can display
    through its visual interface. Facial expressions provide essential non-verbal
    communication cues that enhance human-robot interaction by conveying the
    agent's internal state, reactions, and engagement level.

    Facial expressions are rendered through the agent's display system, which
    may include:

    - **Screen-based avatars**: Animated faces on LCD/OLED displays
    - **LED matrix displays**: Simplified expressions using LED patterns
    - **Robotic face mechanisms**: Physical actuators controlling eyebrows,
      mouth, and other facial features
    - **WebSim interface**: Browser-based avatar visualization

    Attributes
    ----------
    HAPPY : str
        Expresses joy, satisfaction, or positive engagement. Typically shown
        with upturned mouth corners, raised cheeks, and bright eyes. Use when
        greeting users, completing tasks successfully, or responding to
        positive interactions.
    CONFUSED : str
        Expresses uncertainty, puzzlement, or difficulty understanding. Often
        displayed with furrowed brows, tilted head, or asymmetric features.
        Use when encountering ambiguous input or unexpected situations.
    CURIOUS : str
        Expresses interest, inquisitiveness, or active attention. Typically
        shown with wide eyes, raised eyebrows, and forward-leaning posture.
        Use when exploring new environments or encountering novel stimuli.
    EXCITED : str
        Expresses enthusiasm, high energy, or eager anticipation. Often
        displayed with wide eyes, open mouth, and animated movements. Use
        for celebratory moments or when something surprising occurs.
    SAD : str
        Expresses disappointment, empathy, or low energy states. Typically
        shown with downturned mouth, lowered eyebrows, and subdued colors.
        Use when unable to help, saying goodbye, or acknowledging negative
        situations.
    THINK : str
        Expresses contemplation, processing, or deliberation. Often displayed
        with eyes looking upward or to the side, perhaps with a hand-to-chin
        gesture. Use when processing complex requests or searching for
        information.

    Examples
    --------
    >>> from actions.face.interface import FaceAction
    >>> expression = FaceAction.HAPPY
    >>> print(expression.value)
    'happy'

    >>> # Check if an expression is valid
    >>> FaceAction("curious") == FaceAction.CURIOUS
    True

    >>> # Iterate over all available expressions
    >>> for expr in FaceAction:
    ...     print(f"{expr.name}: {expr.value}")
    HAPPY: happy
    CONFUSED: confused
    CURIOUS: curious
    EXCITED: excited
    SAD: sad
    THINK: think

    Notes
    -----
    The visual representation of each expression depends on the connected
    display hardware and avatar system. Expressions are designed to be
    recognizable across different display technologies while maintaining
    consistent emotional meaning.
    """

    HAPPY = "happy"
    CONFUSED = "confused"
    CURIOUS = "curious"
    EXCITED = "excited"
    SAD = "sad"
    THINK = "think"


@dataclass
class FaceInput:
    """
    Input interface for the Face action.

    This dataclass defines the input structure required to trigger a facial
    expression on the agent's display system. The expression is selected from
    the predefined set of FaceAction values, ensuring consistent and
    recognizable visual feedback.

    Parameters
    ----------
    action : FaceAction
        The facial expression to display. Must be one of the valid FaceAction
        enum values: HAPPY, CONFUSED, CURIOUS, EXCITED, SAD, or THINK.

    Examples
    --------
    >>> from actions.face.interface import FaceInput, FaceAction
    >>> # Create an input to show happiness
    >>> happy_face = FaceInput(action=FaceAction.HAPPY)

    >>> # Create an input to show thinking
    >>> thinking_face = FaceInput(action=FaceAction.THINK)

    >>> # Access the expression value
    >>> happy_face.action.value
    'happy'

    >>> # Create expressions for different scenarios
    >>> greeting = FaceInput(action=FaceAction.EXCITED)
    >>> processing = FaceInput(action=FaceAction.THINK)
    >>> error_response = FaceInput(action=FaceAction.CONFUSED)
    """

    action: FaceAction


@dataclass
class Face(Interface[FaceInput, FaceInput]):
    """
    Action interface for displaying facial expressions on the agent's visual system.

    This action controls the agent's facial display, enabling expressive non-verbal
    communication that complements speech and movement. Facial expressions are
    fundamental to natural human-robot interaction, helping users understand the
    agent's state and intentions at a glance.

    The Face action integrates with multiple display backends through connectors:

    - **AvatarProvider**: Renders expressions via Zenoh-connected avatar displays,
      supporting distributed visualization across networked systems.
    - **ROS2**: Publishes face commands to ROS2 topics for integration with
      robot operating system-based platforms.

    Effect: When triggered, the agent updates its visual display to show the
    specified facial expression. The expression typically persists until a new
    expression is set, allowing the agent to maintain consistent visual feedback
    during interactions. Expression transitions may include animations depending
    on the display hardware capabilities.

    Common usage patterns include:

    - **Reactive expressions**: Respond to user input with appropriate emotions
    - **State indication**: Show processing (THINK) while working on tasks
    - **Feedback loops**: Combine with speech for multimodal communication
    - **Ambient presence**: Maintain engaging expressions during idle states

    Parameters
    ----------
    input : FaceInput
        The input containing the desired facial expression to display.
    output : FaceInput
        The output echoing the expression that was displayed. Matches the input
        to confirm the action was processed.

    Examples
    --------
    >>> from actions.face.interface import Face, FaceInput, FaceAction
    >>> # Create a face action to show happiness
    >>> happy_input = FaceInput(action=FaceAction.HAPPY)
    >>> face_action = Face(input=happy_input, output=happy_input)

    >>> # Create a face action for thinking/processing state
    >>> think_input = FaceInput(action=FaceAction.THINK)
    >>> thinking_action = Face(input=think_input, output=think_input)

    See Also
    --------
    actions.emotion.interface.Emotion :
        Action for expressing emotional states (may control additional outputs
        beyond facial display, such as LEDs or sounds).
    actions.speak.interface.Speak :
        Action for verbal communication, often paired with facial expressions.

    Notes
    -----
    **Display Hardware Compatibility**: The Face action is designed to work
    across various display technologies:

    - Screen-based systems receive avatar animation commands
    - LED matrices receive simplified pattern codes
    - Physical robot faces receive servo position commands
    - WebSim renders expressions in the browser interface

    **Expression Persistence**: Expressions remain active until explicitly
    changed. Consider setting appropriate expressions before and after
    speech actions for coherent multimodal communication.

    **Performance Considerations**: Expression updates are lightweight and
    can be called frequently without significant overhead. However, rapid
    switching between expressions may appear jarring to users.

    **Synchronization**: When combining Face with Speak actions, consider
    timing to ensure expressions align with speech content (e.g., show
    HAPPY before delivering good news).
    """

    input: FaceInput
    output: FaceInput
