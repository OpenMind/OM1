from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class FestivalType(str, Enum):
    """
    Enumeration of festival types.
    """

    CHINESE_NEW_YEAR = "chinese_new_year"
    MID_AUTUMN = "mid_autumn"
    DRAGON_BOAT = "dragon_boat"
    NATIONAL_DAY = "national_day"
    CHRISTMAS = "christmas"
    NEW_YEAR = "new_year"
    VALENTINE = "valentine"
    BIRTHDAY = "birthday"
    CUSTOM = "custom"


@dataclass
class FestivalGreetingInput:
    """
    Input interface for the FestivalGreeting action.

    Parameters
    ----------
    festival_type : FestivalType
        The type of festival to greet.
    message : str, optional
        Custom greeting message. If not provided, a default message for the festival will be used.
    recipient_name : str, optional
        Name of the person to greet. If provided, the greeting will be personalized.
    """

    festival_type: FestivalType
    message: str = ""
    recipient_name: str = ""


@dataclass
class FestivalGreeting(Interface[FestivalGreetingInput, FestivalGreetingInput]):
    """
    This action allows the robot to send festival greetings and reminders.

    The robot can greet users on various festivals (Chinese New Year, Mid-Autumn Festival,
    Christmas, New Year, etc.) with personalized messages. It can also remind users about
    upcoming festivals. The greeting can be delivered through text-to-speech, making the
    interaction more natural and engaging.

    Examples:
    - Greet on Chinese New Year: festival_type="chinese_new_year", message="Happy Chinese New Year!"
    - Remind about upcoming festival: festival_type="mid_autumn", message="Mid-Autumn Festival is coming in 3 days, don't forget to prepare mooncakes!"
    - Custom birthday greeting: festival_type="birthday", recipient_name="Alice", message="Happy Birthday!"
    """

    input: FestivalGreetingInput
    output: FestivalGreetingInput

