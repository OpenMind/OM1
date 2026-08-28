from dataclasses import dataclass
from typing import Optional

from actions.base import Interface


@dataclass
class LostAndFoundQueryInput:
    """
    Input payload for querying objects that have been detected
    by the Lost & Found ingestion pipeline.

    Fields
    ------
    action : str
        The name of the object label to query, e.g. "cup", "bottle", "remote".
        If set to a generic value like "all", "everything", "objects", or "*",
        the connector will return / log the latest known sighting for each label.

    room : Optional[str]
        Optional room filter. If provided (e.g. "kitchen", "living_room"),
        the connector will only consider sightings that were tagged with this room.

    limit : int
        Maximum number of sightings to fetch for a specific label.
        When querying "all", this is ignored and only one (latest) sighting per label
        is returned.
    """

    action: str
    room: Optional[str] = None
    limit: int = 5


@dataclass
class LostAndFoundQuery(Interface[LostAndFoundQueryInput, LostAndFoundQueryInput]):
    """
    Query objects detected by the Lost & Found system.

    Typical usage:
    - User says: "Where did you last see my cup?"
      → action = "cup"
    - User says: "What objects can you see around the house?"
      → action = "all"
    - User says: "What objects are in the kitchen?"
      → action = "all", room = "kitchen"

    """

    input: LostAndFoundQueryInput
    output: LostAndFoundQueryInput
