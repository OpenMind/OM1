"""
Tests for actions.selfie.connector.selfie.SelfieConnector.

Covers:
- SelfieConfig defaults / overrides
- Helper methods (_post_json, _get_config, _set_blur, _who_snapshot,
  _wait_any_face, _write_status, _speak, _display_name)
- Response dispatch: one test per result-code branch in _dispatch_response
- Network-error dispatch (HTTP transport failure)
- End-to-end connect() flow: empty name, no-face pre-check, successful
  enrollment, force flag pass-through, busy-retry behavior, blur snapshot
  and restore.

External dependencies (ElevenLabsTTSProvider, IOProvider, requests) are
mocked. Sleep is patched to keep _wait_any_face fast.
"""

from unittest.mock import Mock, patch

import pytest

from actions.selfie.connector.selfie import SelfieConfig, SelfieConnector
from actions.selfie.interface import SelfieInput

# ----- Shared fixtures -----


@pytest.fixture
def mock_dependencies():
    """
    Mock the connector's external collaborators.

    ElevenLabsTTSProvider and IOProvider are patched at the connector module
    level so that their __init__ in SelfieConnector returns Mock instances.
    """
    with (
        patch("actions.selfie.connector.selfie.ElevenLabsTTSProvider") as mock_tts_cls,
        patch("actions.selfie.connector.selfie.IOProvider") as mock_io_cls,
    ):
        mock_tts = Mock()
        mock_io = Mock()
        mock_tts_cls.return_value = mock_tts
        mock_io_cls.return_value = mock_io
        yield mock_tts, mock_io


@pytest.fixture
def connector(mock_dependencies):
    """SelfieConnector with default config and mocked dependencies."""
    return SelfieConnector(SelfieConfig())


# =====================================================================
# SelfieConfig
# =====================================================================


class TestSelfieConfig:
    """SelfieConfig default and override behavior."""

    def test_default_config(self):
        config = SelfieConfig()
        assert config.face_http_base_url == "http://127.0.0.1:6793"
        assert config.face_recent_sec == 1.0
        assert config.poll_ms == 200
        assert config.timeout_sec == 8
        assert config.http_timeout_sec == 5.0

    def test_custom_config(self):
        config = SelfieConfig(
            face_http_base_url="http://custom:9999",
            face_recent_sec=2.0,
            poll_ms=500,
            timeout_sec=30,
            http_timeout_sec=10.0,
        )
        assert config.face_http_base_url == "http://custom:9999"
        assert config.face_recent_sec == 2.0
        assert config.poll_ms == 500
        assert config.timeout_sec == 30
        assert config.http_timeout_sec == 10.0

    def test_partial_override_keeps_other_defaults(self):
        config = SelfieConfig(timeout_sec=20)
        assert config.timeout_sec == 20
        assert config.poll_ms == 200  # default
        assert config.http_timeout_sec == 5.0  # default


# =====================================================================
# SelfieConnector init
# =====================================================================


class TestSelfieConnectorInit:
    """Connector wires config values into instance state."""

    def test_init_sets_fields_from_config(self, connector):
        assert connector.base_url == "http://127.0.0.1:6793"
        assert connector.recent_sec == 1.0
        assert connector.poll_ms == 200
        assert connector.default_timeout == 8
        assert connector.http_timeout == 5.0

    def test_init_attaches_providers_and_clean_state(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        assert connector.elevenlabs_tts_provider is mock_tts
        assert connector.io_provider is mock_io
        assert connector.last_enrolled_id is None
        assert connector.last_match_name is None


# =====================================================================
# Helpers — _post_json, _get_config, _set_blur, _who_snapshot
# =====================================================================


class TestHttpHelpers:
    """HTTP helper methods."""

    def test_post_json_success(self, connector):
        with patch("actions.selfie.connector.selfie.requests") as mock_requests:
            mock_response = Mock()
            mock_response.json.return_value = {"ok": True, "id": "wendy"}
            mock_requests.post.return_value = mock_response

            result = connector._post_json("/selfie", {"id": "wendy", "force": False})
            assert result == {"ok": True, "id": "wendy"}
            mock_requests.post.assert_called_once_with(
                "http://127.0.0.1:6793/selfie",
                json={"id": "wendy", "force": False},
                timeout=5.0,
            )

    def test_post_json_transport_failure_returns_none(self, connector):
        with patch("actions.selfie.connector.selfie.requests") as mock_requests:
            mock_requests.post.side_effect = Exception("Connection refused")
            assert connector._post_json("/selfie", {"id": "x"}) is None

    def test_get_config_returns_dict(self, connector):
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = {"config": {"blur": True}}
            assert connector._get_config() == {"config": {"blur": True}}
            mock_post.assert_called_once_with("/config", {"get": True})

    def test_get_config_none_returns_empty_dict(self, connector):
        with patch.object(connector, "_post_json", return_value=None):
            assert connector._get_config() == {}

    def test_get_config_non_dict_returns_empty(self, connector):
        # Defensive: API misbehaves and returns a list instead of dict
        with patch.object(connector, "_post_json", return_value=["not", "a", "dict"]):
            assert connector._get_config() == {}

    def test_set_blur_on(self, connector):
        with patch.object(connector, "_post_json") as mock_post:
            connector._set_blur(True)
            mock_post.assert_called_once_with("/config", {"set": {"blur": True}})

    def test_set_blur_off(self, connector):
        with patch.object(connector, "_post_json") as mock_post:
            connector._set_blur(False)
            mock_post.assert_called_once_with("/config", {"set": {"blur": False}})

    def test_who_snapshot(self, connector):
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = {"now": ["wendy"], "unknown_now": 1}
            assert connector._who_snapshot() == {"now": ["wendy"], "unknown_now": 1}
            mock_post.assert_called_once_with("/who", {"recent_sec": 1.0})


class TestWaitAnyFace:
    """Pre-check polling behavior."""

    def test_wait_any_face_known_only(self, connector):
        with (
            patch.object(connector, "_who_snapshot") as mock_who,
            patch.object(connector, "sleep"),
        ):
            mock_who.return_value = {"now": ["wendy"], "unknown_now": 0}
            assert connector._wait_any_face(5) is True

    def test_wait_any_face_unknown_only(self, connector):
        with (
            patch.object(connector, "_who_snapshot") as mock_who,
            patch.object(connector, "sleep"),
        ):
            mock_who.return_value = {"now": [], "unknown_now": 1}
            assert connector._wait_any_face(5) is True

    def test_wait_any_face_known_and_unknown(self, connector):
        with (
            patch.object(connector, "_who_snapshot") as mock_who,
            patch.object(connector, "sleep"),
        ):
            mock_who.return_value = {"now": ["wendy"], "unknown_now": 2}
            assert connector._wait_any_face(5) is True

    def test_wait_any_face_timeout(self, connector):
        with (
            patch.object(connector, "_who_snapshot") as mock_who,
            patch.object(connector, "sleep") as mock_sleep,
        ):
            mock_who.return_value = {"now": [], "unknown_now": 0}
            assert connector._wait_any_face(1) is False
            # 1 second / 200ms poll = 5 polls
            assert mock_who.call_count == 5
            # sleep between polls
            assert mock_sleep.called

    def test_wait_any_face_zero_timeout_uses_default(self, connector):
        with (
            patch.object(connector, "_who_snapshot") as mock_who,
            patch.object(connector, "sleep"),
        ):
            mock_who.return_value = {"now": [], "unknown_now": 0}
            connector._wait_any_face(0)
            # default_timeout=8s / 200ms = 40 polls
            assert mock_who.call_count == 40

    def test_wait_any_face_handles_none_response(self, connector):
        """_who_snapshot returning None (network error) shouldn't crash."""
        with (
            patch.object(connector, "_who_snapshot", return_value=None),
            patch.object(connector, "sleep"),
        ):
            assert connector._wait_any_face(1) is False


class TestStatusAndSpeech:
    """_write_status and _speak."""

    def test_write_status_writes_to_selfie_status_key(self, connector, mock_dependencies):
        _, mock_io = mock_dependencies
        connector._write_status("result=success id=wendy samples=3 merged=false")
        mock_io.add_input.assert_called_once()
        args, _kwargs = mock_io.add_input.call_args
        assert args[0] == "SelfieStatus"
        assert args[1] == "result=success id=wendy samples=3 merged=false"
        # third arg is a timestamp (float)
        assert isinstance(args[2], float)

    def test_write_status_swallows_exceptions(self, connector, mock_dependencies):
        _, mock_io = mock_dependencies
        mock_io.add_input.side_effect = RuntimeError("io broken")
        # Should not raise
        connector._write_status("result=foo")

    def test_speak_queues_tts(self, connector, mock_dependencies):
        mock_tts, _ = mock_dependencies
        connector._speak("Nice to meet you, Wendy!")
        mock_tts.add_pending_message.assert_called_once_with("Nice to meet you, Wendy!")

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
        assert SelfieConnector._display_name(internal) == expected


# =====================================================================
# _dispatch_response — one test per branch
# =====================================================================


class TestDispatchResponseSuccess:
    """ok=True path."""

    def test_success_new_enrollment(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        resp = {"ok": True, "id": "wendy", "samples_saved": 3, "merged": False}
        connector._dispatch_response(resp, claimed_id="wendy")

        # State
        assert connector.last_enrolled_id == "wendy"
        assert connector.last_match_name is None

        # SelfieStatus
        status_line = mock_io.add_input.call_args[0][1]
        assert "result=success" in status_line
        assert "id=wendy" in status_line
        assert "samples=3" in status_line
        assert "merged=false" in status_line

        # TTS
        tts_msg = mock_tts.add_pending_message.call_args[0][0]
        assert tts_msg == "Nice to meet you, Wendy! I'll remember you next time."

    def test_success_merged_returning_user(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        resp = {"ok": True, "id": "wendy", "samples_saved": 2, "merged": True}
        connector._dispatch_response(resp, claimed_id="wendy")

        # State: enrolled id set, no mismatched match
        assert connector.last_enrolled_id == "wendy"
        assert connector.last_match_name is None

        # SelfieStatus
        status_line = mock_io.add_input.call_args[0][1]
        assert "result=merged" in status_line
        assert "merged=true" in status_line

        # TTS — neutral, no enrollment language
        tts_msg = mock_tts.add_pending_message.call_args[0][0]
        assert tts_msg == "Welcome back, Wendy!"

    def test_success_id_renamed_by_server(self, connector, mock_dependencies):
        """Server may rename (e.g., dedup → wendy_1); use saved id for display."""
        mock_tts, mock_io = mock_dependencies
        resp = {"ok": True, "id": "wendy_1", "samples_saved": 1, "merged": False}
        connector._dispatch_response(resp, claimed_id="wendy")

        # SelfieStatus uses server-side id
        assert "id=wendy_1" in mock_io.add_input.call_args[0][1]
        # Display strips the _1 suffix
        assert mock_tts.add_pending_message.call_args[0][0].startswith("Nice to meet you, Wendy!")


class TestDispatchResponseErrors:
    """error=... branches."""

    def test_ambiguous_subjects(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "ambiguous_subjects", "n_engaged": 3}, claimed_id="wendy")
        assert mock_io.add_input.call_args[0][1] == "result=ambiguous engaged=3"
        assert "step closer" in mock_tts.add_pending_message.call_args[0][0].lower()
        # state cleared
        assert connector.last_enrolled_id is None
        assert connector.last_match_name is None

    def test_face_belongs_to(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        resp = {"error": "face_belongs_to", "name": "wendy", "sim": 0.72}
        connector._dispatch_response(resp, claimed_id="john")

        # match name tracked for correction flow
        assert connector.last_match_name == "wendy"
        # enrolled id NOT set (this isn't an enrollment)
        assert connector.last_enrolled_id is None

        # SelfieStatus
        status = mock_io.add_input.call_args[0][1]
        assert "result=face_belongs_to" in status
        assert "claimed=john" in status
        assert "matched=wendy" in status
        assert "sim=0.720" in status

        # TTS asks user to clarify, uses display name
        tts = mock_tts.add_pending_message.call_args[0][0]
        assert "Wendy" in tts
        assert "different" in tts.lower()

    def test_no_valid_frames(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "no_valid_frames"}, claimed_id="wendy")
        assert mock_io.add_input.call_args[0][1] == "result=low_quality"
        tts = mock_tts.add_pending_message.call_args[0][0]
        assert "can't see your face clearly" in tts.lower() or "cannot see" in tts.lower()
        assert connector.last_enrolled_id is None

    def test_insufficient_samples(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "insufficient_samples", "got": 0}, claimed_id="wendy")
        assert mock_io.add_input.call_args[0][1] == "result=partial got=0"
        assert "hold still" in mock_tts.add_pending_message.call_args[0][0].lower()

    def test_busy_after_retry(self, connector, mock_dependencies):
        """Dispatch path sees busy only after the connect()-level retry has failed."""
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "busy"}, claimed_id="wendy")
        assert mock_io.add_input.call_args[0][1] == "result=busy retries=1"
        assert mock_tts.add_pending_message.called

    def test_bad_id_no_tts(self, connector, mock_dependencies):
        """bad_id surfaces to LLM only — no TTS noise to the user."""
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response(
            {"error": "bad_id", "detail": "empty"},
            claimed_id="",
        )
        assert mock_io.add_input.call_args[0][1] == "result=bad_id detail=empty"
        mock_tts.add_pending_message.assert_not_called()
        assert connector.last_enrolled_id is None

    def test_recognition_disabled(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "recognition_disabled"}, claimed_id="wendy")
        assert mock_io.add_input.call_args[0][1] == "result=recognition_disabled"
        assert mock_tts.add_pending_message.called

    def test_unknown_error_fallback(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({"error": "something_weird"}, claimed_id="wendy")
        status = mock_io.add_input.call_args[0][1]
        assert status.startswith("result=unknown")
        assert "error=something_weird" in status
        assert "something went wrong" in mock_tts.add_pending_message.call_args[0][0].lower()
        assert connector.last_enrolled_id is None

    def test_missing_error_field_treated_as_unknown(self, connector, mock_dependencies):
        """Malformed response (no ok, no error) falls through to unknown branch."""
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_response({}, claimed_id="wendy")
        status = mock_io.add_input.call_args[0][1]
        assert "result=unknown" in status
        assert "error=unknown" in status


class TestDispatchNetworkError:
    """HTTP transport failure (resp is None)."""

    def test_network_error(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        connector._dispatch_network_error()
        assert mock_io.add_input.call_args[0][1] == "result=network_error"
        assert "lost connection" in mock_tts.add_pending_message.call_args[0][0].lower()
        assert connector.last_enrolled_id is None
        assert connector.last_match_name is None


# =====================================================================
# connect() — end-to-end flow
# =====================================================================


class TestConnect:
    """Top-level connect() orchestration."""

    @pytest.mark.asyncio
    async def test_connect_empty_name_writes_bad_id(self, connector, mock_dependencies):
        mock_tts, mock_io = mock_dependencies
        await connector.connect(SelfieInput(action=""))

        # SelfieStatus written
        mock_io.add_input.assert_called_once()
        assert mock_io.add_input.call_args[0][1] == "result=bad_id detail=empty"
        # No TTS noise for bad_id
        mock_tts.add_pending_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_whitespace_only_name_writes_bad_id(self, connector, mock_dependencies):
        _, mock_io = mock_dependencies
        await connector.connect(SelfieInput(action="   "))
        assert mock_io.add_input.call_args[0][1] == "result=bad_id detail=empty"

    @pytest.mark.asyncio
    async def test_connect_no_face_present(self, connector, mock_dependencies):
        """Pre-check returns no face → low_quality with reason."""
        mock_tts, mock_io = mock_dependencies
        with (
            patch.object(connector, "_get_config", return_value={"config": {"blur": True}}),
            patch.object(connector, "_set_blur") as mock_blur,
            patch.object(connector, "_wait_any_face", return_value=False),
            patch.object(connector, "_post_json") as mock_post,
        ):
            await connector.connect(SelfieInput(action="wendy", timeout_sec=2))

            # status indicates no one present
            status = mock_io.add_input.call_args[0][1]
            assert "result=low_quality" in status
            assert "no_one_present" in status

            # TTS spoke about not seeing anyone
            assert "don't see anyone" in mock_tts.add_pending_message.call_args[0][0].lower()

            # /selfie was NEVER called (pre-check short-circuited)
            assert not any(call.args[0] == "/selfie" for call in mock_post.call_args_list)

            # blur restored to original (True)
            assert mock_blur.call_args_list[-1].args == (True,)

    @pytest.mark.asyncio
    async def test_connect_successful_enrollment_payload(self, connector, mock_dependencies):
        """Successful path: payload uses {id, force}, NOT {action, force}."""
        mock_tts, mock_io = mock_dependencies
        with (
            patch.object(connector, "_get_config", return_value={"config": {"blur": False}}),
            patch.object(connector, "_set_blur") as mock_blur,
            patch.object(connector, "_wait_any_face", return_value=True),
            patch.object(connector, "_post_json") as mock_post,
        ):
            mock_post.return_value = {"ok": True, "id": "wendy", "samples_saved": 3, "merged": False}
            await connector.connect(SelfieInput(action="wendy", timeout_sec=5))

            # Find the /selfie POST call
            selfie_calls = [c for c in mock_post.call_args_list if c.args[0] == "/selfie"]
            assert len(selfie_calls) == 1
            payload = selfie_calls[0].args[1]
            # API uses "id" key, NOT "action" (interface name)
            assert payload == {"id": "wendy", "force": False}

            # TTS played the success line
            tts = mock_tts.add_pending_message.call_args[0][0]
            assert "Nice to meet you, Wendy" in tts

            # Blur was turned off (during) then restored to False (orig)
            blur_values = [c.args[0] for c in mock_blur.call_args_list]
            assert blur_values == [False, False]

    @pytest.mark.asyncio
    async def test_connect_strips_whitespace_in_name(self, connector, mock_dependencies):
        with (
            patch.object(connector, "_get_config", return_value={"config": {"blur": True}}),
            patch.object(connector, "_set_blur"),
            patch.object(connector, "_wait_any_face", return_value=True),
            patch.object(connector, "_post_json") as mock_post,
        ):
            mock_post.return_value = {"ok": True, "id": "wendy", "samples_saved": 1, "merged": False}
            await connector.connect(SelfieInput(action="  wendy  ", timeout_sec=5))
            selfie_calls = [c for c in mock_post.call_args_list if c.args[0] == "/selfie"]
            assert selfie_calls[0].args[1]["id"] == "wendy"

    @pytest.mark.asyncio
    async def test_connect_force_flag_passes_through(self, connector, mock_dependencies):
        """force=True from SelfieInput is forwarded to the API."""
        with (
            patch.object(connector, "_get_config", return_value={"config": {"blur": True}}),
            patch.object(connector, "_set_blur"),
            patch.object(connector, "_wait_any_face", return_value=True),
            patch.object(connector, "_post_json") as mock_post,
        ):
            mock_post.return_value = {"ok": True, "id": "john", "samples_saved": 3, "merged": False}
            await connector.connect(SelfieInput(action="john", force=True))
            selfie_calls = [c for c in mock_post.call_args_list if c.args[0] == "/selfie"]
            assert selfie_calls[0].args[1] == {"id": "john", "force": True}

    @pytest.mark.asyncio
    async def test_connect_force_default_false(self, connector, mock_dependencies):
        """When force not provided, defaults to False."""
        with (
            patch.object(connector, "_get_config", return_value={"config": {"blur": True}}),
            patch.object(connector, "_set_blur"),
            patch.object(connector, "_wait_any_face", return_value=True),
            patch.object(connector, "_post_json") as mock_post,
        ):
            mock_post.return_value = {"ok": True, "id": "wendy", "samples_saved": 1, "merged": False}
            await connector.connect(SelfieInput(action="wendy"))
            selfie_calls = [c for c in mock_post.call_args_list if c.args[0] == "/selfie"]
            assert selfie_calls[0].args[1]["force"] is False

    @pytest.mark.asyncio
    async def test_connect_busy_retries_once(self, connector, mock_dependencies):
        """First /selfie returns busy → wait 1s → retry succeeds."""
        with (
            patch.object(connector, "_get_config", return_value={"config": {"blur": True}}),
            patch.object(connector, "_set_blur"),
            patch.object(connector, "_wait_any_face", return_value=True),
            patch.object(connector, "_post_json") as mock_post,
            patch("actions.selfie.connector.selfie.asyncio.sleep") as mock_async_sleep,
        ):
            # First call busy, second call succeeds
            mock_post.side_effect = [
                {"error": "busy"},
                {"ok": True, "id": "wendy", "samples_saved": 2, "merged": True},
                # any further /config calls during finally
                None,
            ]
            await connector.connect(SelfieInput(action="wendy"))

            # Two /selfie calls
            selfie_calls = [c for c in mock_post.call_args_list if c.args[0] == "/selfie"]
            assert len(selfie_calls) == 2

            # asyncio.sleep(1.0) called between them
            mock_async_sleep.assert_any_call(1.0)

    @pytest.mark.asyncio
    async def test_connect_busy_twice_dispatches_busy(self, connector, mock_dependencies):
        """If retry also returns busy, dispatch_response handles it (result=busy)."""
        _, mock_io = mock_dependencies
        with (
            patch.object(connector, "_get_config", return_value={"config": {"blur": True}}),
            patch.object(connector, "_set_blur"),
            patch.object(connector, "_wait_any_face", return_value=True),
            patch.object(connector, "_post_json") as mock_post,
            patch("actions.selfie.connector.selfie.asyncio.sleep"),
        ):
            mock_post.return_value = {"error": "busy"}
            await connector.connect(SelfieInput(action="wendy"))
            # Final SelfieStatus reflects unresolved busy
            assert any("result=busy" in c.args[1] for c in mock_io.add_input.call_args_list)

    @pytest.mark.asyncio
    async def test_connect_network_error(self, connector, mock_dependencies):
        """/selfie returns None → network_error dispatched."""
        mock_tts, mock_io = mock_dependencies
        with (
            patch.object(connector, "_get_config", return_value={"config": {"blur": True}}),
            patch.object(connector, "_set_blur"),
            patch.object(connector, "_wait_any_face", return_value=True),
            patch.object(connector, "_post_json", return_value=None),
        ):
            await connector.connect(SelfieInput(action="wendy"))
            status_lines = [c.args[1] for c in mock_io.add_input.call_args_list]
            assert any(line == "result=network_error" for line in status_lines)

    @pytest.mark.asyncio
    async def test_connect_restores_blur_even_on_exception(self, connector, mock_dependencies):
        """If something blows up mid-flow, blur is still restored."""
        with (
            patch.object(connector, "_get_config", return_value={"config": {"blur": True}}),
            patch.object(connector, "_set_blur") as mock_blur,
            patch.object(connector, "_wait_any_face", side_effect=RuntimeError("boom")),
        ):
            with pytest.raises(RuntimeError):
                await connector.connect(SelfieInput(action="wendy"))
            # First call: off (False). Last call (in finally): restore original True.
            blur_values = [c.args[0] for c in mock_blur.call_args_list]
            assert blur_values[0] is False
            assert blur_values[-1] is True

    @pytest.mark.asyncio
    async def test_connect_face_belongs_to_updates_match_state(self, connector, mock_dependencies):
        """Full flow exercising the face_belongs_to dispatch path."""
        with (
            patch.object(connector, "_get_config", return_value={"config": {"blur": True}}),
            patch.object(connector, "_set_blur"),
            patch.object(connector, "_wait_any_face", return_value=True),
            patch.object(connector, "_post_json") as mock_post,
        ):
            mock_post.return_value = {"error": "face_belongs_to", "name": "wendy", "sim": 0.81}
            await connector.connect(SelfieInput(action="john"))
            assert connector.last_match_name == "wendy"
            assert connector.last_enrolled_id is None
