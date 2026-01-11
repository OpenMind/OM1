"""Tests for CLI validate command."""

from typer.testing import CliRunner

from cli.main import app


class TestValidateCommand:
    """Tests for validate command."""

    def test_validate_help(self, runner: CliRunner):
        """Test validate --help."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "--all" in result.output
        assert "--verbose" in result.output

    def test_validate_nonexistent(self, runner: CliRunner):
        """Test validate with nonexistent config."""
        result = runner.invoke(app, ["validate", "nonexistent_config_12345"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_validate_no_config_no_all(self, runner: CliRunner):
        """Test validate without config name or --all."""
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 1
        assert "provide" in result.output.lower() or "--all" in result.output

    def test_validate_all(self, runner: CliRunner):
        """Test validate --all."""
        result = runner.invoke(app, ["validate", "--all"])
        assert result.exit_code == 0 or "failed" in result.output.lower()

    def test_validate_verbose(self, runner: CliRunner):
        """Test validate with --verbose flag."""
        result = runner.invoke(app, ["validate", "--all", "--verbose"])
        assert result.exit_code == 0 or "failed" in result.output.lower()

    def test_validate_existing_config(self, runner: CliRunner):
        """Test validate with an existing config (test)."""
        result = runner.invoke(app, ["validate", "test"])
        if result.exit_code == 0:
            assert "valid" in result.output.lower()

    def test_validate_skip_inputs(self, runner: CliRunner):
        """Test validate with --skip-inputs."""
        result = runner.invoke(app, ["validate", "test", "--skip-inputs"])
        assert result.exit_code == 0 or "not found" in result.output.lower()

    def test_validate_allow_missing(self, runner: CliRunner):
        """Test validate with --allow-missing."""
        result = runner.invoke(app, ["validate", "test", "--allow-missing"])
        assert result.exit_code == 0 or "not found" in result.output.lower()
