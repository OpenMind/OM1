from dataclasses import dataclass

from actions.base import Interface


@dataclass
class LGThinQInput:
    """Input for LG ThinQ actions. The action field contains the command string."""

    action: str


@dataclass
class LGThinQ(Interface[LGThinQInput, LGThinQInput]):
    """
    Control LG Air Conditioner based on room temperature.
    Commands: set cooling mode, set heating mode, turn off, idle.
    """

    input: LGThinQInput
    output: LGThinQInput
