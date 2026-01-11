"""Tests for CLI test command."""

from typer.testing import CliRunner

from cli.main import app


class TestTestCommand:
    """Tests for test command."""

    def test_test_help(self, runner: CliRunner):
        """Test test --help."""
        result = runner.invoke(app, ["test", "--help"])
        assert result.exit_code == 0
        assert "--unit" in result.output
        assert "--integration" in result.output
        assert "--coverage" in result.output
        assert "--all" in result.output

    def test_test_nonexistent_path(self, runner: CliRunner):
        """Test test with nonexistent path."""
        result = runner.invoke(app, ["test", "nonexistent_path_12345"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_test_verbose(self, runner: CliRunner):
        """Test test --verbose option exists."""
        result = runner.invoke(app, ["test", "--help"])
        assert "--verbose" in result.output

    def test_test_markers(self, runner: CliRunner):
        """Test test --markers option exists."""
        result = runner.invoke(app, ["test", "--help"])
        assert "--markers" in result.output

    def test_test_keyword(self, runner: CliRunner):
        """Test test --keyword option exists."""
        result = runner.invoke(app, ["test", "--help"])
        assert "--keyword" in result.output

    def test_test_parallel(self, runner: CliRunner):
        """Test test --parallel option exists."""
        result = runner.invoke(app, ["test", "--help"])
        assert "--parallel" in result.output

    def test_test_fail_fast(self, runner: CliRunner):
        """Test test --fail-fast option exists."""
        result = runner.invoke(app, ["test", "--help"])
        assert "--fail-fast" in result.output
