from zenoh_msgs.idl.status_msgs import (
    AIStatusRequest,
    AIStatusResponse,
    ASRText,
    AudioStatus,
    AvatarFaceRequest,
    AvatarFaceResponse,
    CameraStatus,
    ChargingStatus,
    ConfigRequest,
    ConfigResponse,
    ModeStatusRequest,
    ModeStatusResponse,
    PersonGreetingStatus,
    TTSStatusRequest,
    TTSStatusResponse,
)
from zenoh_msgs.idl.std_msgs import Header, String, Time


def make_header(frame_id: str = "map") -> Header:
    """Helper to create a Header."""
    return Header(stamp=Time(sec=1, nanosec=0), frame_id=frame_id)


class TestAudioStatus:
    """Tests for the AudioStatus dataclass."""

    def test_create_audio_status(self):
        """Test AudioStatus creation."""
        audio = AudioStatus(
            header=make_header(),
            status_mic=AudioStatus.STATUS_MIC.READY.value,
            status_speaker=AudioStatus.STATUS_SPEAKER.ACTIVE.value,
            sentence_to_speak=String(data="Hello"),
        )
        assert audio.status_mic == AudioStatus.STATUS_MIC.READY.value
        assert audio.status_speaker == AudioStatus.STATUS_SPEAKER.ACTIVE.value
        assert audio.sentence_to_speak.data == "Hello"

    def test_audio_status_mic_enum(self):
        """Test AudioStatus STATUS_MIC enum values."""
        assert AudioStatus.STATUS_MIC.DISABLED.value == 0
        assert AudioStatus.STATUS_MIC.READY.value == 1
        assert AudioStatus.STATUS_MIC.ACTIVE.value == 2
        assert AudioStatus.STATUS_MIC.UNKNOWN.value == 3


class TestCameraStatus:
    """Tests for the CameraStatus dataclass."""

    def test_create_camera_status(self):
        """Test CameraStatus creation."""
        cs = CameraStatus(
            header=make_header(),
            status=CameraStatus.STATUS.ENABLED.value,
        )
        assert cs.status == CameraStatus.STATUS.ENABLED.value

    def test_camera_status_enum(self):
        """Test CameraStatus STATUS enum values."""
        assert CameraStatus.STATUS.DISABLED.value == 0
        assert CameraStatus.STATUS.ENABLED.value == 1


class TestAIStatusRequest:
    """Tests for the AIStatusRequest dataclass."""

    def test_create_ai_status_request(self):
        """Test AIStatusRequest creation."""
        req = AIStatusRequest(
            header=make_header(),
            request_id=String(data="req-001"),
            code=AIStatusRequest.Code.ENABLED.value,
        )
        assert req.request_id.data == "req-001"
        assert req.code == AIStatusRequest.Code.ENABLED.value

    def test_ai_status_request_enum(self):
        """Test AIStatusRequest Code enum values."""
        assert AIStatusRequest.Code.DISABLED.value == 0
        assert AIStatusRequest.Code.ENABLED.value == 1
        assert AIStatusRequest.Code.STATUS.value == 2


class TestAIStatusResponse:
    """Tests for the AIStatusResponse dataclass."""

    def test_create_ai_status_response(self):
        """Test AIStatusResponse creation."""
        resp = AIStatusResponse(
            header=make_header(),
            request_id=String(data="req-001"),
            code=AIStatusResponse.Code.ENABLED.value,
            status=String(data="ok"),
        )
        assert resp.code == AIStatusResponse.Code.ENABLED.value
        assert resp.status.data == "ok"


class TestModeStatusRequest:
    """Tests for the ModeStatusRequest dataclass."""

    def test_create_mode_status_request(self):
        """Test ModeStatusRequest creation."""
        req = ModeStatusRequest(
            header=make_header(),
            request_id=String(data="req-002"),
            code=ModeStatusRequest.Code.SWITCH_MODE.value,
            mode=String(data="autonomous"),
        )
        assert req.code == ModeStatusRequest.Code.SWITCH_MODE.value
        assert req.mode.data == "autonomous"

    def test_mode_status_request_default_mode(self):
        """Test ModeStatusRequest default mode value."""
        req = ModeStatusRequest(
            header=make_header(),
            request_id=String(data="req-003"),
            code=ModeStatusRequest.Code.STATUS.value,
        )
        assert req.mode == String("")


class TestModeStatusResponse:
    """Tests for the ModeStatusResponse dataclass."""

    def test_create_mode_status_response(self):
        """Test ModeStatusResponse creation."""
        resp = ModeStatusResponse(
            header=make_header(),
            request_id=String(data="req-002"),
            code=ModeStatusResponse.Code.SUCCESS.value,
            current_mode=String(data="autonomous"),
            message=String(data="switched"),
        )
        assert resp.code == ModeStatusResponse.Code.SUCCESS.value
        assert resp.current_mode.data == "autonomous"

    def test_mode_status_response_enum(self):
        """Test ModeStatusResponse Code enum values."""
        assert ModeStatusResponse.Code.SUCCESS.value == 0
        assert ModeStatusResponse.Code.FAILURE.value == 1
        assert ModeStatusResponse.Code.UNKNOWN.value == 2


class TestTTSStatusRequest:
    """Tests for the TTSStatusRequest dataclass."""

    def test_create_tts_status_request(self):
        """Test TTSStatusRequest creation."""
        req = TTSStatusRequest(
            header=make_header(),
            request_id=String(data="req-004"),
            code=TTSStatusRequest.Code.ENABLED.value,
        )
        assert req.code == TTSStatusRequest.Code.ENABLED.value


class TestTTSStatusResponse:
    """Tests for the TTSStatusResponse dataclass."""

    def test_create_tts_status_response(self):
        """Test TTSStatusResponse creation."""
        resp = TTSStatusResponse(
            header=make_header(),
            request_id=String(data="req-004"),
            code=TTSStatusResponse.Code.ENABLED.value,
            status=String(data="active"),
        )
        assert resp.code == TTSStatusResponse.Code.ENABLED.value
        assert resp.status.data == "active"


class TestASRText:
    """Tests for the ASRText dataclass."""

    def test_create_asr_text(self):
        """Test ASRText creation."""
        asr = ASRText(header=make_header(), text="hello world")
        assert asr.text == "hello world"

    def test_asr_text_empty(self):
        """Test ASRText with empty text."""
        asr = ASRText(header=make_header(), text="")
        assert asr.text == ""


class TestAvatarFaceRequest:
    """Tests for the AvatarFaceRequest dataclass."""

    def test_create_avatar_face_request(self):
        """Test AvatarFaceRequest creation."""
        req = AvatarFaceRequest(
            header=make_header(),
            request_id=String(data="req-005"),
            code=AvatarFaceRequest.Code.SWITCH_FACE.value,
            face_text=String(data="happy"),
        )
        assert req.code == AvatarFaceRequest.Code.SWITCH_FACE.value
        assert req.face_text.data == "happy"


class TestAvatarFaceResponse:
    """Tests for the AvatarFaceResponse dataclass."""

    def test_create_avatar_face_response(self):
        """Test AvatarFaceResponse creation."""
        resp = AvatarFaceResponse(
            header=make_header(),
            request_id=String(data="req-005"),
            code=AvatarFaceResponse.Code.ACTIVE.value,
            message=String(data="face switched"),
        )
        assert resp.code == AvatarFaceResponse.Code.ACTIVE.value
        assert resp.message.data == "face switched"

    def test_avatar_face_response_enum(self):
        """Test AvatarFaceResponse Code enum values."""
        assert AvatarFaceResponse.Code.ACTIVE.value == 0
        assert AvatarFaceResponse.Code.INACTIVE.value == 1
        assert AvatarFaceResponse.Code.UNKNOWN.value == 2


class TestConfigRequest:
    """Tests for the ConfigRequest dataclass."""

    def test_create_config_request(self):
        """Test ConfigRequest creation."""
        req = ConfigRequest(
            header=make_header(),
            request_id=String(data="req-006"),
            config=String(data='{"key": "value"}'),
        )
        assert req.request_id.data == "req-006"
        assert req.config.data == '{"key": "value"}'

    def test_config_request_default_config(self):
        """Test ConfigRequest default config value."""
        req = ConfigRequest(
            header=make_header(),
            request_id=String(data="req-007"),
        )
        assert req.config == String("")


class TestConfigResponse:
    """Tests for the ConfigResponse dataclass."""

    def test_create_config_response(self):
        """Test ConfigResponse creation."""
        resp = ConfigResponse(
            header=make_header(),
            request_id=String(data="req-006"),
            config=String(data='{"key": "value"}'),
            message=String(data="success"),
        )
        assert resp.config.data == '{"key": "value"}'
        assert resp.message.data == "success"


class TestChargingStatus:
    """Tests for the ChargingStatus dataclass."""

    def test_create_charging_status(self):
        """Test ChargingStatus creation."""
        cs = ChargingStatus(
            header=make_header(),
            code=ChargingStatus.Code.CHARGING.value,
            status=String(data="charging"),
        )
        assert cs.code == ChargingStatus.Code.CHARGING.value
        assert cs.status.data == "charging"

    def test_charging_status_enum(self):
        """Test ChargingStatus Code enum values."""
        assert ChargingStatus.Code.DISCHARGING.value == 0
        assert ChargingStatus.Code.CHARGING.value == 1
        assert ChargingStatus.Code.ENROUTE_CHARGING.value == 2
        assert ChargingStatus.Code.FULLY_CHARGED.value == 3


class TestPersonGreetingStatus:
    """Tests for the PersonGreetingStatus dataclass."""

    def test_create_person_greeting_status(self):
        """Test PersonGreetingStatus creation."""
        pgs = PersonGreetingStatus(
            header=make_header(),
            request_id=String(data="req-008"),
            status=PersonGreetingStatus.STATUS.APPROACHING.value,
            message=String(data="person detected"),
        )
        assert pgs.status == PersonGreetingStatus.STATUS.APPROACHING.value
        assert pgs.message.data == "person detected"

    def test_person_greeting_status_enum(self):
        """Test PersonGreetingStatus STATUS enum values."""
        assert PersonGreetingStatus.STATUS.APPROACHING.value == 0
        assert PersonGreetingStatus.STATUS.APPROACHED.value == 1
        assert PersonGreetingStatus.STATUS.SWITCH.value == 2
        assert PersonGreetingStatus.STATUS.CONVERSATION.value == 3
