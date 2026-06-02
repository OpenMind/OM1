"""
Tests for actions.correct_identity.connector.correct_identity.CorrectIdentityConnector.

Covers:
- CorrectIdentityConfig defaults / overrides
- Helper methods (_post_json, _write_status, _speak, _display_name)
- Response dispatch: one test per result-code branch in _dispatch_response
- Network-error dispatch (HTTP transport failure)
- End-to-end connect() flow: local validation (bad_id when ids missing,
  same_id no-op when ids equal after normalization), payload shape, and
  error response handling

External dependencies (ElevenLabsTTSProvider, IOProvider, requests) are
mocked.
"""

from unittest.mock import Mock, patch

import pytest

from actions.correct_identity.connector.correct_identity import (
    CorrectIdentityConfig,
    CorrectIdentityConnector,
)
from actions.correct_identity.interface import CorrectIdentityInput

# ----- Shared fixtures -----


@pytest.fixture
def mock_dependencies():
    """
    Mock the connector's external collaborators.

    ElevenLabsTTSProvider and IOProvider are patched at the connector module
    level so that their __init__ in CorrectIdentityConnector returns Mock
    instances.
    """
    with (
        patch("actions.correct_identity.connector.correct_identity.ElevenLabsTTSProvider") as mock_tts_cls,
        patch("actions.correct_identity.connector.correct_identity.IOProvider") as mock_io_cls,
    ):
        mock_tts = Mock()
        mock_io = Mock()
        mock_tts_cls.return_value = mock_tts
        mock_io_cls.return_value = mock_io
        yield mock_tts, mock_io


@pytest.fixture
def connector(mock_dependencies):
    """CorrectIdentityConnector with default config and mocked dependencies."""
    return CorrectIdentityConnector(CorrectIdentityConfig())


# =====================================================================
# CorrectIdentityConfig
# =====================================================================


class TestCorrectIdentityConfig:
    """CorrectIdentityConfig default and override behavior."""

    def test_default_config(self):
        config = CorrectIdentityConfig()
        assert config.face_http_base_url == "http://127.0.0.1:6793"
        assert config.http_timeout_sec == 5.0

    def test_custom_config(self):
        config = CorrectIdentityConfig(
            face_http_base_url="http://custom:9999",
            http_timeout_sec=10.0,
        )
        assert config.face_http_base_url == "http://custom:9999"
        assert config.http_timeout_sec == 10.0

    def test_partial_override_keeps_other_defaults(self):
        config = CorrectIdentityConfig(http_timeout_sec=20.0)
        assert config.http_timeout_sec == 20.0
        assert config.face_http_base_url == "http://127.0.0.1:6793"  # default


# =====================================================================
# CorrectIdentityConnector init
# =====================================================================


class TestCorrectIdentityConnectorInit:
    """Connector wires config values into instance state."""

    def test_init_sets_fields_from_config(self, connector):
        assert connector.base_url == "http://127.0.0.1:6793"
        assert connector.http_timeout == 5.0

    def test_init_attaches_providers(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        assert connector.elevenlabs_tts_provider is mock_tts
        assert connector.io_provider is mock_io


# =====================================================================
# HTTP helpers
# =====================================================================


class TestHttpHelpers:
    """HTTP helper methods."""

    def test_post_json_success(self, connector):
        with patch("actions.correct_identity.connector.correct_identity.requests") as mock_requests:
            mock_response = Mock()
            mock_response.json.return_value = {"ok": True, "moved": 3}
            mock_requests.post.return_value = mock_response

            result = connector._post_json("/gallery/move_samples", {"from_id": "wendy", "to_id": "wendi"})
            assert result == {"ok": True, "moved": 3}
            mock_requests.post.assert_called_once_with(
                "http://127.0.0.1:6793/gallery/move_samples",
                json={"from_id": "wendy", "to_id": "wendi"},
                timeout=5.0,
            )

    def test_post_json_transport_failure_returns_none(self, connector):
        with patch("actions.correct_identity.connector.correct_identity.requests") as mock_requests:
            mock_requests.post.side_effect = Exception("Connection refused")
            assert connector._post_json("/gallery/move_samples", {}) is None


# =====================================================================
# _write_status / _speak
# =====================================================================


class TestStatusAndSpeech:
    """_write_status and _speak."""

    def test_write_status_writes_to_shared_selfie_status_key(self, connector, mock_dependencies):
        """
        correct_identity writes to the SHARED 'SelfieStatus' channel,
        the same key used by selfie and forget_last. The LLM
        disambiguates by reading the result=... prefix.
        """
        _, mock_io = mock_dependencies
        connector._write_status("result=success from=wendy to=wendi moved=3 from_removed=true")
        mock_io.add_input.assert_called_once()
        args, _kwargs = mock_io.add_input.call_args
        assert args[0] == "SelfieStatus"
        assert args[1] == ("result=success from=wendy to=wendi moved=3 from_removed=true")
        assert isinstance(args[2], float)

    def test_write_status_swallows_exceptions(self, connector, mock_dependencies):
        _, mock_io = mock_dependencies
        mock_io.add_input.side_effect = RuntimeError("io broken")
        # Should not raise
        connector._write_status("result=foo")

    def test_speak_queues_tts(self, connector, mock_dependencies):
        mock_tts, _ = mock_dependencies
        connector._speak("Got it, I've updated your name to Wendy.")
        mock_tts.add_pending_message.assert_called_once_with("Got it, I've updated your name to Wendy.")

    def test_speak_empty_string_noop(self, connector, mock_dependencies):
        mock_tts, _ = mock_dependencies
        connector._speak("")
        mock_tts.add_pending_message.assert_not_called()

    def test_speak_none_noop(self, connector, mock_dependencies):
        mock_tts, _ = mock_dependencies
        connector._speak(None)
        mock_tts.add_pending_message.assert_not_called()

    def test_speak_swallows_exceptions(self, connector, mock_dependencies):
        mock_tts, _ = mock_dependencies
        mock_tts.add_pending_message.side_effect = RuntimeError("tts broken")
        # Should not raise
        connector._speak("hello")


# =====================================================================
# _display_name
# =====================================================================


class TestDisplayName:
    """Static helper that converts internal id → human-friendly name."""

    @pytest.mark.parametrize(
        "internal,expected",
        [
            ("wendy", "Wendy"),
            ("wendy_1", "Wendy"),  # strip dedup suffix
            ("wendy_42", "Wendy"),  # arbitrary digit suffix
            ("jerin-peter", "Jerin Peter"),  # dash → space + title
            ("jerin-peter_3", "Jerin Peter"),  # both transformations
            ("li-xiaohong", "Li Xiaohong"),
            ("first_last", "First Last"),  # internal underscore → space
            ("MIXED-Case", "Mixed Case"),
            ("a", "A"),
        ],
    )
    def test_display_name_cases(self, internal, expected):
        assert CorrectIdentityConnector._display_name(internal) == expected


# =====================================================================
# _dispatch_response — one test per branch
# =====================================================================


class TestDispatchResponseSuccess:
    """ok=True path."""

    def test_success_full_response(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        resp = {"ok": True, "moved": 3, "from_removed": True}
        connector._dispatch_response(resp, from_id="wendi", to_id="wendy")

        # SelfieStatus
        status = mock_io.add_input.call_args[0][1]
        assert "result=success" in status
        assert "from=wendi" in status
        assert "to=wendy" in status
        assert "moved=3" in status
        assert "from_removed=true" in status

        # TTS uses display name of to_id
        assert mock_tts.add_pending_message.call_args[0][0] == "Got it, I've updated your name to Wendy."

    def test_success_from_not_removed(self, connector, mock_dependencies):
        """E.g. when to_id already existed and from_id stays around."""
        _, mock_io = mock_dependencies
        resp = {"ok": True, "moved": 2, "from_removed": False}
        connector._dispatch_response(resp, from_id="wendi", to_id="wendy")
        status = mock_io.add_input.call_args[0][1]
        assert "from_removed=false" in status

    def test_success_tts_uses_to_id_display_name(self, connector, mock_dependencies):
        """to_id 'jerin-peter_1' → TTS 'Jerin Peter'."""
        mock_tts, _ = mock_dependencies
        resp = {"ok": True, "moved": 1, "from_removed": True}
        connector._dispatch_response(resp, from_id="jared", to_id="jerin-peter_1")
        assert mock_tts.add_pending_message.call_args[0][0] == "Got it, I've updated your name to Jerin Peter."

    def test_success_with_missing_fields_uses_defaults(self, connector, mock_dependencies):
        """Minimal API response — defaults applied without crashing."""
        _, mock_io = mock_dependencies
        connector._dispatch_response({"ok": True}, from_id="a", to_id="b")
        status = mock_io.add_input.call_args[0][1]
        assert "moved=0" in status
        assert "from_removed=false" in status


class TestDispatchResponseErrors:
    """error=... branches."""

    def test_no_recent_enrollment(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "no_recent_enrollment"}, from_id="a", to_id="b")
        assert mock_io.add_input.call_args[0][1] == "result=no_recent_enrollment"
        assert "too much time" in mock_tts.add_pending_message.call_args[0][0].lower()

    def test_stale_enrollment(self, connector, mock_dependencies):
        """stale_enrollment shares the same TTS as no_recent_enrollment."""
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "stale_enrollment"}, from_id="a", to_id="b")
        assert mock_io.add_input.call_args[0][1] == "result=stale_enrollment"
        assert "too much time" in mock_tts.add_pending_message.call_args[0][0].lower()

    def test_bad_id_from_api_no_tts(self, connector, mock_dependencies):
        """bad_id surfaces to LLM only — no TTS noise to the user."""
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response(
            {"error": "bad_id", "detail": "to_id contains space"},
            from_id="a",
            to_id="b c",
        )
        status = mock_io.add_input.call_args[0][1]
        assert "result=bad_id" in status
        assert "detail=to_id contains space" in status
        mock_tts.add_pending_message.assert_not_called()

    def test_same_id_from_api_no_tts(self, connector, mock_dependencies):
        """API-side same_id (e.g., from canonicalization) → silent."""
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "same_id"}, from_id="wendy", to_id="wendy_alias")
        # Uses the connector's from_id for the status line
        assert mock_io.add_input.call_args[0][1] == "result=same_id id=wendy"
        mock_tts.add_pending_message.assert_not_called()

    def test_no_safe_files(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "no_safe_files"}, from_id="a", to_id="b")
        assert mock_io.add_input.call_args[0][1] == "result=no_safe_files"
        assert "couldn't find" in mock_tts.add_pending_message.call_args[0][0].lower()

    def test_recognition_disabled(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "recognition_disabled"}, from_id="a", to_id="b")
        assert mock_io.add_input.call_args[0][1] == "result=recognition_disabled"
        assert mock_tts.add_pending_message.called

    def test_unknown_error_fallback(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "something_weird"}, from_id="a", to_id="b")
        status = mock_io.add_input.call_args[0][1]
        assert status.startswith("result=unknown")
        assert "error=something_weird" in status
        assert "something went wrong" in mock_tts.add_pending_message.call_args[0][0].lower()

    def test_missing_error_field_treated_as_unknown(self, connector, mock_dependencies):
        """Malformed response (no ok, no error) → unknown branch."""
        _, mock_io = mock_dependencies
        connector._dispatch_response({}, from_id="a", to_id="b")
        status = mock_io.add_input.call_args[0][1]
        assert "result=unknown" in status
        assert "error=unknown" in status


class TestDispatchNetworkError:
    """HTTP transport failure (resp is None)."""

    def test_network_error(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response(None, from_id="a", to_id="b")
        assert mock_io.add_input.call_args[0][1] == "result=network_error"
        assert "trouble updating" in mock_tts.add_pending_message.call_args[0][0].lower()


# =====================================================================
# connect() — end-to-end flow
# =====================================================================


class TestConnect:
    """Top-level connect() orchestration including local validation."""

    @pytest.mark.asyncio
    async def test_connect_empty_from_id_writes_bad_id_silent(self, connector, mock_dependencies):
        """Empty from_id is a local-validation error; no API call, no TTS."""
        mock_tts, mock_io = mock_dependencies
        with patch.object(connector, "_post_json") as mock_post:
            await connector.connect(CorrectIdentityInput(from_id="", to_id="wendy"))
            mock_post.assert_not_called()
            status = mock_io.add_input.call_args[0][1]
            assert "result=bad_id" in status
            mock_tts.add_pending_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_empty_to_id_writes_bad_id_silent(self, connector, mock_dependencies):
        """Empty to_id is a local-validation error; no API call, no TTS."""
        mock_tts, mock_io = mock_dependencies
        with patch.object(connector, "_post_json") as mock_post:
            await connector.connect(CorrectIdentityInput(from_id="wendy", to_id=""))
            mock_post.assert_not_called()
            assert "result=bad_id" in mock_io.add_input.call_args[0][1]
            mock_tts.add_pending_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_whitespace_only_ids_treated_as_bad_id(self, connector, mock_dependencies):
        """'   ' strips to '' → bad_id."""
        _, mock_io = mock_dependencies
        with patch.object(connector, "_post_json") as mock_post:
            await connector.connect(CorrectIdentityInput(from_id="   ", to_id="wendy"))
            mock_post.assert_not_called()
            assert "result=bad_id" in mock_io.add_input.call_args[0][1]

    @pytest.mark.asyncio
    async def test_connect_same_id_after_normalization_short_circuits(self, connector, mock_dependencies):
        """
        Raw inputs that differ only by case/whitespace ('Wendy' vs '  wendy  ')
        normalize to the same id → no-op, no API call, no TTS.
        """
        mock_tts, mock_io = mock_dependencies
        with patch.object(connector, "_post_json") as mock_post:
            await connector.connect(CorrectIdentityInput(from_id="Wendy", to_id="  wendy  "))
            mock_post.assert_not_called()
            assert mock_io.add_input.call_args[0][1] == "result=same_id id=wendy"
            mock_tts.add_pending_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_normalizes_payload_to_lower_strip(self, connector, mock_dependencies):
        """Mixed-case + whitespace input normalizes before hitting the API."""
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = {
                "ok": True,
                "moved": 2,
                "from_removed": True,
            }
            await connector.connect(CorrectIdentityInput(from_id="  Wendi  ", to_id="WENDY"))

            path, body = mock_post.call_args[0]
            assert path == "/gallery/move_samples"
            assert body == {"from_id": "wendi", "to_id": "wendy"}

    @pytest.mark.asyncio
    async def test_connect_success_dispatches_success(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = {
                "ok": True,
                "moved": 4,
                "from_removed": True,
            }
            await connector.connect(CorrectIdentityInput(from_id="wendi", to_id="wendy"))

            status = mock_io.add_input.call_args[0][1]
            assert "result=success" in status
            assert "from=wendi" in status
            assert "to=wendy" in status
            assert "moved=4" in status
            # TTS uses display name of to_id
            assert "Wendy" in mock_tts.add_pending_message.call_args[0][0]

    @pytest.mark.asyncio
    async def test_connect_api_bad_id_silent(self, connector, mock_dependencies):
        """If API returns bad_id (e.g. invalid chars), connector stays silent."""
        mock_tts, mock_io = mock_dependencies
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = {
                "error": "bad_id",
                "detail": "non-ascii",
            }
            await connector.connect(CorrectIdentityInput(from_id="wendy", to_id="wendi"))
            status = mock_io.add_input.call_args[0][1]
            assert "result=bad_id" in status
            assert "detail=non-ascii" in status
            mock_tts.add_pending_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_no_recent_enrollment(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = {"error": "no_recent_enrollment"}
            await connector.connect(CorrectIdentityInput(from_id="wendi", to_id="wendy"))
            assert mock_io.add_input.call_args[0][1] == "result=no_recent_enrollment"
            assert mock_tts.add_pending_message.called

    @pytest.mark.asyncio
    async def test_connect_network_error(self, connector, mock_dependencies):
        """/gallery/move_samples returns None → network_error dispatched."""
        _, mock_io = mock_dependencies
        with patch.object(connector, "_post_json", return_value=None):
            await connector.connect(CorrectIdentityInput(from_id="wendi", to_id="wendy"))
            assert mock_io.add_input.call_args[0][1] == "result=network_error"
