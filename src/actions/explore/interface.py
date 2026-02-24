from dataclasses import dataclass
from typing import Optional

from actions.base import Interface


@dataclass
class ExploreInput:
    """
    Input payload for the explore action.

    Parameters
    ----------
    action : str
        The action to perform. Must be either "explore" or "stop explore".
    duration : Optional[int]
        Optional time limit for exploration in seconds.
        If None, exploration continues until no frontiers remain or manually stopped.
    return_to_start : bool
        Whether the robot should return to its starting position after exploration ends.
        Defaults to True.

    Examples
    --------
    - User says: "Start exploring" → action = "explore"
    - User says: "Explore for 60 seconds" → action = "explore", duration = 60
    - User says: "Stop exploring" → action = "stop explore"
    - User says: "Explore but don't come back" → action = "explore", return_to_start = False
    """

    action: str
    duration: Optional[int] = None
    return_to_start: bool = True


@dataclass
class Explore(Interface[ExploreInput, ExploreInput]):
    """
    Explore the environment autonomously using frontier-based exploration.

    Use action = "explore" to start and action = "stop explore" to stop.
    Optionally specify a duration (in seconds) and whether to return to the starting position.
    """

    input: ExploreInput
    output: ExploreInput
