"""Tests for CLI main module."""

from typer.testing import CliRunner

from cli.main import app


class TestCLIMain:
    """Tests for main CLI entry point."""

    def test_help(self, runner: CliRunner):
        """Test --help flag."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "OM1" in result.output
        assert "run" in result.output
        assert "list" in result.output

    def test_version(self, runner: CliRunner):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_no_args_shows_help(self, runner: CliRunner):
        """Test that no arguments shows help."""
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage" in result.output or "Commands" in result.output

    def test_run_command_exists(self, runner: CliRunner):
        """Test that run command exists."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "configuration" in result.output.lower()

    def test_list_command_exists(self, runner: CliRunner):
        """Test that list command exists."""
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0

    def test_info_command_exists(self, runner: CliRunner):
        """Test that info command exists."""
        result = runner.invoke(app, ["info", "--help"])
        assert result.exit_code == 0

    def test_validate_command_exists(self, runner: CliRunner):
        """Test that validate command exists."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0

    def test_test_command_exists(self, runner: CliRunner):
        """Test that test command exists."""
        result = runner.invoke(app, ["test", "--help"])
        assert result.exit_code == 0

    def test_lint_command_exists(self, runner: CliRunner):
        """Test that lint command exists."""
        result = runner.invoke(app, ["lint", "--help"])
        assert result.exit_code == 0

    def test_init_command_exists(self, runner: CliRunner):
        """Test that init command exists."""
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0

    def test_modes_command_exists(self, runner: CliRunner):
        """Test that modes command exists."""
        result = runner.invoke(app, ["modes", "--help"])
        assert result.exit_code == 0
