from dataclasses import dataclass
from enum import Enum

from pycdr2 import IdlStruct
from pycdr2.types import int8

from .std_msgs import Header, String


@dataclass
class AudioStatus(IdlStruct, typename="AudioStatus"):
    """Audio status message."""

    class STATUS_MIC(Enum):
        """Mic status code enum for AudioStatus."""

        DISABLED = 0
        READY = 1
        ACTIVE = 2
        UNKNOWN = 3

    class STATUS_SPEAKER(Enum):
        """Speaker status code enum for AudioStatus."""

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
    """Camera status message."""

    class STATUS(Enum):
        """Code enum for CameraStatus."""

        DISABLED = 0
        ENABLED = 1

    header: Header
    status: int8


@dataclass
class AIStatusRequest(IdlStruct, typename="AIStatusRequest"):
    """AI status request message."""

    class Code(Enum):
        """Code enum for AIStatusRequest."""

        DISABLED = 0
        ENABLED = 1
        STATUS = 2

    header: Header
    request_id: String
    code: int8


@dataclass
class AIStatusResponse(IdlStruct, typename="AIStatusResponse"):
    """AI status response message."""

    class Code(Enum):
        """Code enum for AIStatusResponse."""

        DISABLED = 0
        ENABLED = 1
        UNKNOWN = 2

    header: Header
    request_id: String
    code: int8
    status: String


@dataclass
class ModeStatusRequest(IdlStruct, typename="ModeStatusRequest"):
    """Mode status request message."""

    class Code(Enum):
        """Code enum for ModeStatusRequest."""

        SWITCH_MODE = 0
        STATUS = 1

    header: Header
    request_id: String
    code: int8
    mode: String = String("")  # Target mode for SWITCH_MODE, ignored for STATUS


@dataclass
class ModeStatusResponse(IdlStruct, typename="ModeStatusResponse"):
    """Mode status response message."""

    class Code(Enum):
        """Code enum for ModeStatusResponse."""

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
    """TTS status request message."""

    class Code(Enum):
        """Code enum for TTSStatusRequest."""

        DISABLED = 0
        ENABLED = 1
        STATUS = 2

    header: Header
    request_id: String
    code: int8


@dataclass
class TTSStatusResponse(IdlStruct, typename="TTSStatusResponse"):
    """TTS status response message."""

    class Code(Enum):
        """Code enum for TTSStatusResponse."""

        DISABLED = 0
        ENABLED = 1
        UNKNOWN = 2

    header: Header
    request_id: String
    code: int8
    status: String


@dataclass
class ASRText(IdlStruct, typename="ASRText"):
    """ASR text message."""

    header: Header
    text: str


@dataclass
class AvatarFaceRequest(IdlStruct, typename="AvatarFaceRequest"):
    """Avatar face request message."""

    class Code(Enum):
        """Code enum for AvatarFaceRequest."""

        SWITCH_FACE = 0
        STATUS = 1

    header: Header
    request_id: String
    code: int8
    face_text: String


@dataclass
class AvatarFaceResponse(IdlStruct, typename="AvatarFaceResponse"):
    """Avatar face response message."""

    class Code(Enum):
        """Code enum for AvatarFaceResponse."""

        ACTIVE = 0
        INACTIVE = 1
        UNKNOWN = 2

    header: Header
    request_id: String
    code: int8
    message: String


@dataclass
class ConfigRequest(IdlStruct, typename="ConfigRequest"):
    """Request message for configuration requests."""

    header: Header
    request_id: String
    config: String = String("")  # ignored for GET_CONFIG


@dataclass
class ConfigResponse(IdlStruct, typename="ConfigResponse"):
    """Response message for configuration requests."""

    header: Header
    request_id: String
    config: String
    message: String
