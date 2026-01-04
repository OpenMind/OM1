import json
from typing import cast

import json5
from zenoh import Publisher

from providers.config_provider import ConfigProvider
from zenoh_msgs import String


class DummyPublisher:
    def __init__(self):
        self.published = []

    def put(self, payload):
        self.published.append(payload)


def test_get_config_snapshot_returns_empty_when_file_missing(tmp_path):
    provider = ConfigProvider()

    provider.config_path = str(tmp_path / ".runtime.json5")

    result = provider._get_config_snapshot()

    assert isinstance(result, dict)
    assert result == {}


def test_get_config_snapshot_reads_valid_json5(tmp_path):
    provider = ConfigProvider()

    provider.config_path = str(tmp_path / ".runtime.json5")

    data = {"name": "test", "version": "1.0.0"}

    with open(provider.config_path, "w") as f:
        json5.dump(data, f)

    result = provider._get_config_snapshot()

    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["version"] == "1.0.0"


def test_send_config_response_uses_publisher(tmp_path):
    provider = ConfigProvider()

    provider.config_path = str(tmp_path / ".runtime.json5")

    with open(provider.config_path, "w") as f:
        json.dump({"foo": "bar"}, f)

    dummy = DummyPublisher()
    provider.config_response_publisher = cast(Publisher, dummy)

    provider._send_config_response(String("req-1"))

    assert len(dummy.published) == 1


def test_send_error_response_uses_publisher():
    provider = ConfigProvider()

    dummy = DummyPublisher()
    provider.config_response_publisher = cast(Publisher, dummy)

    provider._send_error_response(String("req-2"), "something failed")

    assert len(dummy.published) == 1


def test_get_config_snapshot_returns_empty_on_invalid_json(tmp_path):
    provider = ConfigProvider()

    provider.config_path = str(tmp_path / ".runtime.json5")

    with open(provider.config_path, "w") as f:
        f.write("this is not json")

    result = provider._get_config_snapshot()

    assert isinstance(result, dict)
    assert result == {}