from dataclasses import dataclass

from pycdr2 import Enum, IdlStruct
from pycdr2.types import int8

from .std_msgs import Header, String


@dataclass
class AudioStatus(IdlStruct, typename="AudioStatus"):
    class STATUS_MIC(Enum):
        DISABLED = 0
        READY = 1
        ACTIVE = 2
        UNKNOWN = 3

    class STATUS_SPEAKER(Enum):
        DISABLED = 0
        READY = 1
        ACTIVE = 2
        UNKNOWN = 3

    header: Header
    status_mic: int8
    status_speaker: int8
    sentence_to_speak: String


@dataclass
class CameraStatus(IdlStruct, typename="CameraStatus"):
    class STATUS(Enum):
        DISABLED = 0
        ENABLED = 1

    header: Header
    status: int8


@dataclass
class AIControlStatus(IdlStruct, typename="AIControlStatus"):
    class STATUS(Enum):
        DISABLED = 0
        ENABLED = 1

    header: Header
    status: int8
