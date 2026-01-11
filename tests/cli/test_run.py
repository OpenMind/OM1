"""Tests for CLI run command."""

from typer.testing import CliRunner

from cli.main import app


class TestRunCommand:
    """Tests for run command."""

    def test_run_help(self, runner: CliRunner):
        """Test run --help."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--hot-reload" in result.output
        assert "--log-level" in result.output
        assert "--check-interval" in result.output

    def test_run_nonexistent(self, runner: CliRunner):
        """Test run with nonexistent config."""
        result = runner.invoke(app, ["run", "nonexistent_config_12345"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_run_log_levels(self, runner: CliRunner):
        """Test that log level options are documented."""
        result = runner.invoke(app, ["run", "--help"])
        assert "DEBUG" in result.output or "log" in result.output.lower()
