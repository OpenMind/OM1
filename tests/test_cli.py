import pytest

from cli import _resolve_config_path


@pytest.mark.parametrize("config_name", ["", "   "])
def test_resolve_config_path_rejects_blank(config_name):
    with pytest.raises(FileNotFoundError, match="Configuration name is empty"):
        _resolve_config_path(config_name)


def test_resolve_config_path_strips_whitespace(tmp_path):
    config_path = tmp_path / "example.json5"
    config_path.write_text("{}")

    resolved = _resolve_config_path(f" {config_path} ")

    assert resolved == str(config_path.resolve())
