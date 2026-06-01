"""
Tests for actions.forget_last.connector.forget_last.ForgetLastConnector.

Covers:
- ForgetLastConfig defaults / overrides
- Helper methods (_post_json, _write_status, _speak)
- Response dispatch: one test per result-code branch in _dispatch_response,
  with particular attention to the id_mismatch branch which threads the
  user-requested id into the status line for LLM context
- Network-error dispatch (HTTP transport failure)
- End-to-end connect() flow: with/without id, payload normalization
  (lowercase + strip), and error response handling

External dependencies (ElevenLabsTTSProvider, IOProvider, requests) are
mocked.
"""

from unittest.mock import Mock, patch

import pytest

from actions.forget_last.connector.forget_last import (
    ForgetLastConfig,
    ForgetLastConnector,
)
from actions.forget_last.interface import ForgetLastInput

# ----- Shared fixtures -----


@pytest.fixture
def mock_dependencies():
    """
    Mock the connector's external collaborators.

    ElevenLabsTTSProvider and IOProvider are patched at the connector module
    level so that their __init__ in ForgetLastConnector returns Mock instances.
    """
    with (
        patch("actions.forget_last.connector.forget_last.ElevenLabsTTSProvider") as mock_tts_cls,
        patch("actions.forget_last.connector.forget_last.IOProvider") as mock_io_cls,
    ):
        mock_tts = Mock()
        mock_io = Mock()
        mock_tts_cls.return_value = mock_tts
        mock_io_cls.return_value = mock_io
        yield mock_tts, mock_io


@pytest.fixture
def connector(mock_dependencies):
    """ForgetLastConnector with default config and mocked dependencies."""
    return ForgetLastConnector(ForgetLastConfig())


# =====================================================================
# ForgetLastConfig
# =====================================================================


class TestForgetLastConfig:
    """ForgetLastConfig default and override behavior."""

    def test_default_config(self):
        config = ForgetLastConfig()
        assert config.face_http_base_url == "http://127.0.0.1:6793"
        assert config.http_timeout_sec == 5.0

    def test_custom_config(self):
        config = ForgetLastConfig(
            face_http_base_url="http://custom:9999",
            http_timeout_sec=10.0,
        )
        assert config.face_http_base_url == "http://custom:9999"
        assert config.http_timeout_sec == 10.0

    def test_partial_override_keeps_other_defaults(self):
        config = ForgetLastConfig(http_timeout_sec=20.0)
        assert config.http_timeout_sec == 20.0
        assert config.face_http_base_url == "http://127.0.0.1:6793"  # default


# =====================================================================
# ForgetLastConnector init
# =====================================================================


class TestForgetLastConnectorInit:
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
        with patch("actions.forget_last.connector.forget_last.requests") as mock_requests:
            mock_response = Mock()
            mock_response.json.return_value = {"ok": True, "id": "wendy"}
            mock_requests.post.return_value = mock_response

            result = connector._post_json("/gallery/forget_last", {"id": "wendy"})
            assert result == {"ok": True, "id": "wendy"}
            mock_requests.post.assert_called_once_with(
                "http://127.0.0.1:6793/gallery/forget_last",
                json={"id": "wendy"},
                timeout=5.0,
            )

    def test_post_json_transport_failure_returns_none(self, connector):
        with patch("actions.forget_last.connector.forget_last.requests") as mock_requests:
            mock_requests.post.side_effect = Exception("Connection refused")
            assert connector._post_json("/gallery/forget_last", {}) is None


# =====================================================================
# _write_status / _speak
# =====================================================================


class TestStatusAndSpeech:
    """_write_status and _speak."""

    def test_write_status_writes_to_shared_selfie_status_key(self, connector, mock_dependencies):
        """
        forget_last writes to the SHARED 'SelfieStatus' channel,
        the same key used by selfie and correct_identity. The LLM
        disambiguates by reading the result=... prefix.
        """
        _, mock_io = mock_dependencies
        connector._write_status("result=success id=wendy files_deleted=3 identity_removed=true")
        mock_io.add_input.assert_called_once()
        args, _kwargs = mock_io.add_input.call_args
        assert args[0] == "SelfieStatus"
        assert args[1] == ("result=success id=wendy files_deleted=3 identity_removed=true")
        assert isinstance(args[2], float)

    def test_write_status_swallows_exceptions(self, connector, mock_dependencies):
        _, mock_io = mock_dependencies
        mock_io.add_input.side_effect = RuntimeError("io broken")
        # Should not raise
        connector._write_status("result=foo")

    def test_speak_queues_tts(self, connector, mock_dependencies):
        mock_tts, _ = mock_dependencies
        connector._speak("OK, I've forgotten that one. Let's try again.")
        mock_tts.add_pending_message.assert_called_once_with("OK, I've forgotten that one. Let's try again.")

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
# _dispatch_response — one test per branch
# =====================================================================


class TestDispatchResponseSuccess:
    """ok=True path."""

    def test_success_full_response(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        resp = {
            "ok": True,
            "id": "wendy",
            "files_deleted": 3,
            "identity_removed": True,
        }
        connector._dispatch_response(resp, requested_id="wendy")

        # SelfieStatus
        status = mock_io.add_input.call_args[0][1]
        assert "result=success" in status
        assert "id=wendy" in status
        assert "files_deleted=3" in status
        assert "identity_removed=true" in status

        # TTS
        assert mock_tts.add_pending_message.call_args[0][0] == "OK, I've forgotten that one. Let's try again."

    def test_success_identity_not_removed(self, connector, mock_dependencies):
        """Only samples deleted, identity retained (e.g. had multiple samples)."""
        _, mock_io = mock_dependencies
        resp = {
            "ok": True,
            "id": "wendy",
            "files_deleted": 1,
            "identity_removed": False,
        }
        connector._dispatch_response(resp, requested_id="wendy")
        status = mock_io.add_input.call_args[0][1]
        assert "identity_removed=false" in status

    def test_success_with_missing_fields_uses_defaults(self, connector, mock_dependencies):
        """API returns minimal response — defaults applied without crashing."""
        _, mock_io = mock_dependencies
        connector._dispatch_response({"ok": True}, requested_id=None)
        status = mock_io.add_input.call_args[0][1]
        assert "files_deleted=0" in status
        assert "identity_removed=false" in status


class TestDispatchResponseErrors:
    """error=... branches."""

    def test_no_recent_enrollment(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "no_recent_enrollment"}, requested_id=None)
        assert mock_io.add_input.call_args[0][1] == "result=no_recent_enrollment"
        assert "nothing recent" in mock_tts.add_pending_message.call_args[0][0].lower()

    def test_stale_enrollment(self, connector, mock_dependencies):
        """stale_enrollment shares the same TTS as no_recent_enrollment."""
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "stale_enrollment"}, requested_id=None)
        assert mock_io.add_input.call_args[0][1] == "result=stale_enrollment"
        assert "nothing recent" in mock_tts.add_pending_message.call_args[0][0].lower()

    def test_id_mismatch_includes_requested(self, connector, mock_dependencies):
        """
        Critical regression test for the vulture-driven fix:
        requested_id MUST be threaded into the SelfieStatus line so the LLM
        has both sides of the disagreement (what user asked vs server state).
        """
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response(
            {"error": "id_mismatch", "detail": "last_was=david"},
            requested_id="Wendy",  # raw, mixed-case
        )
        status = mock_io.add_input.call_args[0][1]
        # requested is normalized (lowercased, stripped)
        assert "result=id_mismatch" in status
        assert "requested=wendy" in status
        assert "detail=last_was=david" in status
        # TTS
        assert "doesn't match" in mock_tts.add_pending_message.call_args[0][0].lower()

    def test_id_mismatch_none_requested_id(self, connector, mock_dependencies):
        """If requested_id is None, status still has 'requested=' (empty value)."""
        _, mock_io = mock_dependencies
        connector._dispatch_response(
            {"error": "id_mismatch", "detail": "last_was=david"},
            requested_id=None,
        )
        status = mock_io.add_input.call_args[0][1]
        assert "requested=" in status
        assert "detail=last_was=david" in status

    def test_id_mismatch_normalizes_whitespace(self, connector, mock_dependencies):
        """Surrounding whitespace and casing are normalized in the status line."""
        _, mock_io = mock_dependencies
        connector._dispatch_response(
            {"error": "id_mismatch", "detail": ""},
            requested_id="  WENDY  ",
        )
        assert "requested=wendy" in mock_io.add_input.call_args[0][1]

    def test_no_safe_files(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "no_safe_files"}, requested_id="wendy")
        assert mock_io.add_input.call_args[0][1] == "result=no_safe_files"
        assert "couldn't find" in mock_tts.add_pending_message.call_args[0][0].lower()

    def test_recognition_disabled(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "recognition_disabled"}, requested_id=None)
        assert mock_io.add_input.call_args[0][1] == "result=recognition_disabled"
        assert mock_tts.add_pending_message.called

    def test_unknown_error_fallback(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "something_weird"}, requested_id=None)
        status = mock_io.add_input.call_args[0][1]
        assert status.startswith("result=unknown")
        assert "error=something_weird" in status
        assert "something went wrong" in mock_tts.add_pending_message.call_args[0][0].lower()

    def test_missing_error_field_treated_as_unknown(self, connector, mock_dependencies):
        """Malformed response (no ok, no error) → unknown branch."""
        _, mock_io = mock_dependencies
        connector._dispatch_response({}, requested_id=None)
        status = mock_io.add_input.call_args[0][1]
        assert "result=unknown" in status
        assert "error=unknown" in status


class TestDispatchNetworkError:
    """HTTP transport failure (resp is None)."""

    def test_network_error(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response(None, requested_id=None)
        assert mock_io.add_input.call_args[0][1] == "result=network_error"
        assert "couldn't undo" in mock_tts.add_pending_message.call_args[0][0].lower()


# =====================================================================
# connect() — end-to-end flow
# =====================================================================


class TestConnect:
    """Top-level connect() orchestration."""

    @pytest.mark.asyncio
    async def test_connect_no_id_sends_empty_body(self, connector, mock_dependencies):
        """No id provided → body is {} (API uses its 60s TTL to decide)."""
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = {
                "ok": True,
                "id": "david",
                "files_deleted": 3,
                "identity_removed": True,
            }
            await connector.connect(ForgetLastInput())

            mock_post.assert_called_once()
            path, body = mock_post.call_args[0]
            assert path == "/gallery/forget_last"
            assert body == {}  # no id_check → body stays empty

    @pytest.mark.asyncio
    async def test_connect_with_id_normalizes_to_lower_strip(self, connector, mock_dependencies):
        """Raw 'Wendy' / '  wendy  ' / 'WENDY' all normalize to 'wendy' in body."""
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = {
                "ok": True,
                "id": "wendy",
                "files_deleted": 1,
                "identity_removed": True,
            }
            await connector.connect(ForgetLastInput(id="  Wendy  "))

            body = mock_post.call_args[0][1]
            assert body == {"id": "wendy"}

    @pytest.mark.asyncio
    async def test_connect_empty_string_id_omitted_from_body(self, connector, mock_dependencies):
        """Falsy id_check (empty string) is skipped — body stays empty."""
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = {
                "ok": True,
                "id": "x",
                "files_deleted": 0,
                "identity_removed": False,
            }
            await connector.connect(ForgetLastInput(id=""))
            body = mock_post.call_args[0][1]
            assert "id" not in body

    @pytest.mark.asyncio
    async def test_connect_success_dispatches_success(self, connector, mock_dependencies):
        _, mock_io = mock_dependencies
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = {
                "ok": True,
                "id": "david",
                "files_deleted": 4,
                "identity_removed": True,
            }
            await connector.connect(ForgetLastInput(id="david"))
            status = mock_io.add_input.call_args[0][1]
            assert "result=success" in status
            assert "id=david" in status
            assert "files_deleted=4" in status
            assert "identity_removed=true" in status

    @pytest.mark.asyncio
    async def test_connect_id_mismatch_threads_requested_id(self, connector, mock_dependencies):
        """
        End-to-end version of the vulture-fix regression test:
        a raw 'Wendy' input must surface as 'requested=wendy' in the
        status line when the API reports id_mismatch.
        """
        _, mock_io = mock_dependencies
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = {
                "error": "id_mismatch",
                "detail": "last_was=david",
            }
            await connector.connect(ForgetLastInput(id="Wendy"))

            status = mock_io.add_input.call_args[0][1]
            assert "result=id_mismatch" in status
            assert "requested=wendy" in status
            assert "detail=last_was=david" in status

    @pytest.mark.asyncio
    async def test_connect_no_recent_enrollment(self, connector, mock_dependencies):
        _, mock_io = mock_dependencies
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = {"error": "no_recent_enrollment"}
            await connector.connect(ForgetLastInput())
            assert mock_io.add_input.call_args[0][1] == "result=no_recent_enrollment"

    @pytest.mark.asyncio
    async def test_connect_network_error(self, connector, mock_dependencies):
        """/gallery/forget_last returns None → network_error dispatched."""
        _, mock_io = mock_dependencies
        with patch.object(connector, "_post_json", return_value=None):
            await connector.connect(ForgetLastInput(id="wendy"))
            assert mock_io.add_input.call_args[0][1] == "result=network_error"
