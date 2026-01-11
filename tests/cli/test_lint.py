"""Tests for CLI lint command."""

from typer.testing import CliRunner

from cli.main import app


class TestLintCommand:
    """Tests for lint command."""

    def test_lint_help(self, runner: CliRunner):
        """Test lint --help."""
        result = runner.invoke(app, ["lint", "--help"])
        assert result.exit_code == 0
        assert "--fix" in result.output
        assert "--format" in result.output
        assert "--type-check" in result.output

    def test_lint_nonexistent_path(self, runner: CliRunner):
        """Test lint with nonexistent path."""
        result = runner.invoke(app, ["lint", "nonexistent_path_12345"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_lint_check_mode(self, runner: CliRunner):
        """Test lint --check mode."""
        result = runner.invoke(app, ["lint", "--check"])
        assert result.exit_code == 0 or result.exit_code == 1

    def test_lint_all_checks(self, runner: CliRunner):
        """Test lint --all."""
        result = runner.invoke(app, ["lint", "--all"])
        assert result.exit_code == 0 or result.exit_code == 1

    def test_lint_verbose(self, runner: CliRunner):
        """Test lint --verbose."""
        result = runner.invoke(app, ["lint", "--verbose", "--check"])
        assert result.exit_code == 0 or result.exit_code == 1
