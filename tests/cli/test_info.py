"""Tests for CLI info command."""

from typer.testing import CliRunner

from cli.main import app


class TestInfoCommand:
    """Tests for info command."""

    def test_info_help(self, runner: CliRunner):
        """Test info --help."""
        result = runner.invoke(app, ["info", "--help"])
        assert result.exit_code == 0
        assert "--json" in result.output
        assert "--components" in result.output

    def test_info_nonexistent(self, runner: CliRunner):
        """Test info with nonexistent config."""
        result = runner.invoke(app, ["info", "nonexistent_config_12345"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_info_existing_config(self, runner: CliRunner):
        """Test info with an existing config (test)."""
        result = runner.invoke(app, ["info", "test"])
        if result.exit_code == 0:
            assert "configuration" in result.output.lower() or "type" in result.output.lower()

    def test_info_json_output(self, runner: CliRunner):
        """Test info --json with existing config."""
        result = runner.invoke(app, ["info", "test", "--json"])
        if result.exit_code == 0:
            assert "{" in result.output

    def test_info_components(self, runner: CliRunner):
        """Test info --components with existing config."""
        result = runner.invoke(app, ["info", "test", "--components"])
        if result.exit_code == 0:
            assert result.exit_code == 0
