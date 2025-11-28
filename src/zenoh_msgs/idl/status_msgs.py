from dataclasses import dataclass
from enum import Enum

from pycdr2 import IdlStruct
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
class AIStatusRequest(IdlStruct, typename="AIStatusRequest"):
    class Code(Enum):
        DISABLED = 0
        ENABLED = 1
        STATUS = 2

    header: Header
    request_id: String
    code: int8


@dataclass
class AIStatusResponse(IdlStruct, typename="AIStatusResponse"):
    class Code(Enum):
        DISABLED = 0
        ENABLED = 1
        UNKNOWN = 2

    header: Header
    request_id: String
    code: int8
    status: String


@dataclass
class ModeStatusRequest(IdlStruct, typename="ModeStatusRequest"):
    class Code(Enum):
        SWITCH_MODE = 0
        STATUS = 1

    header: Header
    request_id: String
    code: int8
    mode: String = String("")  # Target mode for SWITCH_MODE, ignored for STATUS


@dataclass
class ModeStatusResponse(IdlStruct, typename="ModeStatusResponse"):
    class Code(Enum):
        SUCCESS = 0
        FAILURE = 1
        UNKNOWN = 2

    header: Header
    request_id: String
    code: int8
    current_mode: String
    message: String


@dataclass
class TTSStatusRequest(IdlStruct, typename="TTSStatusRequest"):
    class Code(Enum):
        DISABLED = 0
        ENABLED = 1
        STATUS = 2

    header: Header
    request_id: String
    code: int8


@dataclass
class TTSStatusResponse(IdlStruct, typename="TTSStatusResponse"):
    class Code(Enum):
        DISABLED = 0
        ENABLED = 1
        UNKNOWN = 2

    header: Header
    request_id: String
    code: int8
    status: String


@dataclass
class ASRText(IdlStruct, typename="ASRText"):
    header: Header
    text: str


@dataclass
class AvatarFaceRequest(IdlStruct, typename="AvatarFaceRequest"):
    class Code(Enum):
        SWITCH_FACE = 0
        STATUS = 1

    header: Header
    request_id: String
    code: int8
    face_text: String


@dataclass
class AvatarFaceResponse(IdlStruct, typename="AvatarFaceResponse"):
    class Code(Enum):
        ACTIVE = 0
        INACTIVE = 1
        UNKNOWN = 2

    header: Header
    request_id: String
    code: int8
    message: String
