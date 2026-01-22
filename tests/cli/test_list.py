"""Tests for CLI list command."""

from typer.testing import CliRunner

from cli.main import app


class TestListCommand:
    """Tests for list command."""

    def test_list_help(self, runner: CliRunner):
        """Test list --help."""
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0
        assert (
            "mode-aware" in result.output.lower() or "standard" in result.output.lower()
        )

    def test_list_default(self, runner: CliRunner):
        """Test list command with default options."""
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0

    def test_list_format_simple(self, runner: CliRunner):
        """Test list --format simple."""
        result = runner.invoke(app, ["list", "--format", "simple"])
        assert result.exit_code == 0

    def test_list_format_json(self, runner: CliRunner):
        """Test list --format json."""
        result = runner.invoke(app, ["list", "--format", "json"])
        assert result.exit_code == 0

    def test_list_mode_aware_filter(self, runner: CliRunner):
        """Test list --mode-aware filter."""
        result = runner.invoke(app, ["list", "--mode-aware"])
        assert result.exit_code == 0

    def test_list_standard_filter(self, runner: CliRunner):
        """Test list --standard filter."""
        result = runner.invoke(app, ["list", "--standard"])
        assert result.exit_code == 0
