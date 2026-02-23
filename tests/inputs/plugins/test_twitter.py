import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from inputs.plugins.twitter import TwitterInput, TwitterSensorConfig


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.twitter.IOProvider") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def twitter_input(mock_io_provider):
    config = TwitterSensorConfig(poll_interval=10.0)
    t = TwitterInput(config)
    t.io_provider = mock_io_provider
    return t


class TestTwitterSensorConfig:

    def test_default_values(self):
        config = TwitterSensorConfig()
        assert config.query == "What's new in AI and technology?"
        assert config.poll_interval == 60.0

    def test_custom_values(self):
        config = TwitterSensorConfig(query="Python news", poll_interval=120.0)
        assert config.query == "Python news"
        assert config.poll_interval == 120.0


class TestTwitterInput:

    def test_init_with_config(self, mock_io_provider):
        config = TwitterSensorConfig(query="robotics", poll_interval=30.0)
        t = TwitterInput(config)
        assert t.query == "robotics"
        assert t.poll_interval == 30.0
        assert t.descriptor_for_LLM == "TwitterInput CONTEXT"
        assert t.messages == []

    def test_init_without_config(self, mock_io_provider):
        t = TwitterInput()
        assert t.query == "What's new in AI and technology?"
        assert t.poll_interval == 60.0


class TestTwitterInputInitSession:

    @pytest.mark.asyncio
    async def test_creates_session_when_none(self, twitter_input):
        twitter_input.session = None
        with patch("inputs.plugins.twitter.aiohttp.ClientSession") as mock_client:
            mock_client.return_value = MagicMock()
            await twitter_input._init_session()
            assert twitter_input.session is not None
            mock_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_if_session_already_exists(self, twitter_input):
        existing_session = MagicMock()
        twitter_input.session = existing_session
        await twitter_input._init_session()
        assert twitter_input.session is existing_session


class TestTwitterInputPoll:

    def _make_mock_response(self, status, data=None, text=None):
        mock_response = AsyncMock()
        mock_response.status = status
        if data is not None:
            mock_response.json = AsyncMock(return_value=data)
        if text is not None:
            mock_response.text = AsyncMock(return_value=text)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        return mock_response

    def _set_mock_session(self, twitter_input, response=None, side_effect=None):
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=response, side_effect=side_effect)
        twitter_input.session = mock_session

    @pytest.mark.asyncio
    async def test_returns_none_before_interval(self, twitter_input):
        twitter_input._last_poll_time = time.time()
        result = await twitter_input._poll()
        assert result is None

    @pytest.mark.asyncio
    async def test_fetches_after_interval(self, twitter_input):
        mock_data = {"results": [{"content": {"text": "AI news today"}}]}
        twitter_input._last_poll_time = time.time() - 20.0

        with patch.object(twitter_input, "_init_session", new_callable=AsyncMock):
            self._set_mock_session(
                twitter_input, self._make_mock_response(200, data=mock_data)
            )
            result = await twitter_input._poll()
            assert result == mock_data

    @pytest.mark.asyncio
    async def test_returns_none_on_api_error(self, twitter_input):
        twitter_input._last_poll_time = time.time() - 20.0

        with patch.object(twitter_input, "_init_session", new_callable=AsyncMock):
            self._set_mock_session(
                twitter_input,
                self._make_mock_response(500, text="Internal Server Error"),
            )
            result = await twitter_input._poll()
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self, twitter_input):
        twitter_input._last_poll_time = time.time() - 20.0

        with patch.object(twitter_input, "_init_session", new_callable=AsyncMock):
            self._set_mock_session(twitter_input, side_effect=asyncio.TimeoutError)
            result = await twitter_input._poll()
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_client_error(self, twitter_input):
        twitter_input._last_poll_time = time.time() - 20.0

        with patch.object(twitter_input, "_init_session", new_callable=AsyncMock):
            self._set_mock_session(twitter_input, side_effect=aiohttp.ClientError)
            result = await twitter_input._poll()
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_unexpected_error(self, twitter_input):
        twitter_input._last_poll_time = time.time() - 20.0

        with patch.object(twitter_input, "_init_session", new_callable=AsyncMock):
            self._set_mock_session(
                twitter_input, side_effect=RuntimeError("unexpected")
            )
            result = await twitter_input._poll()
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_if_session_is_none(self, twitter_input):
        twitter_input._last_poll_time = time.time() - 20.0

        with patch.object(twitter_input, "_init_session", new_callable=AsyncMock):
            twitter_input.session = None
            result = await twitter_input._poll()
            assert result is None


class TestTwitterInputRawToText:

    @pytest.mark.asyncio
    async def test_none_input_returns_none(self, twitter_input):
        result = await twitter_input._raw_to_text(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_success_multi_document(self, twitter_input):
        raw = {
            "results": [
                {"content": {"text": "First document"}},
                {"content": {"text": "Second document"}},
            ]
        }
        result = await twitter_input._raw_to_text(raw)
        assert result is not None
        assert "First document" in result.message
        assert "Second document" in result.message

    @pytest.mark.asyncio
    async def test_empty_results_returns_none(self, twitter_input):
        result = await twitter_input._raw_to_text({"results": []})
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_results_key_returns_none(self, twitter_input):
        result = await twitter_input._raw_to_text({"other_key": "value"})
        assert result is None

    @pytest.mark.asyncio
    async def test_filters_empty_text(self, twitter_input):
        raw = {
            "results": [
                {"content": {"text": "Valid content"}},
                {"content": {"text": ""}},
                {"content": {}},
            ]
        }
        result = await twitter_input._raw_to_text(raw)
        assert result is not None
        assert "Valid content" in result.message

    @pytest.mark.asyncio
    async def test_invalid_input_returns_none(self, twitter_input):
        result = await twitter_input._raw_to_text({"results": "invalid_not_a_list"})
        assert result is None

    @pytest.mark.asyncio
    async def test_appends_to_messages(self, twitter_input):
        raw = {"results": [{"content": {"text": "Some context"}}]}
        await twitter_input.raw_to_text(raw)
        assert len(twitter_input.messages) == 1
        assert "Some context" in twitter_input.messages[0].message

    @pytest.mark.asyncio
    async def test_none_does_not_append(self, twitter_input):
        await twitter_input.raw_to_text(None)
        assert len(twitter_input.messages) == 0


class TestTwitterInputFormattedBuffer:

    def test_empty_returns_none(self, twitter_input):
        assert twitter_input.formatted_latest_buffer() is None

    @pytest.mark.asyncio
    async def test_formats_output_correctly(self, twitter_input):
        raw = {"results": [{"content": {"text": "Context data"}}]}
        await twitter_input.raw_to_text(raw)

        result = twitter_input.formatted_latest_buffer()
        assert result is not None
        assert "TwitterInput CONTEXT" in result
        assert "Context data" in result
        assert "// START" in result
        assert "// END" in result

    @pytest.mark.asyncio
    async def test_clears_messages_after_format(self, twitter_input):
        raw = {"results": [{"content": {"text": "Context data"}}]}
        await twitter_input.raw_to_text(raw)
        assert len(twitter_input.messages) == 1

        twitter_input.formatted_latest_buffer()
        assert len(twitter_input.messages) == 0

    @pytest.mark.asyncio
    async def test_calls_io_provider(self, twitter_input):
        raw = {"results": [{"content": {"text": "Context data"}}]}
        await twitter_input.raw_to_text(raw)

        twitter_input.formatted_latest_buffer()
        twitter_input.io_provider.add_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_latest_only(self, twitter_input):
        raw1 = {"results": [{"content": {"text": "Old context"}}]}
        raw2 = {"results": [{"content": {"text": "New context"}}]}
        await twitter_input.raw_to_text(raw1)
        await twitter_input.raw_to_text(raw2)

        result = twitter_input.formatted_latest_buffer()
        assert "New context" in result
