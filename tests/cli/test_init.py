"""Tests for CLI init command."""

from pathlib import Path

from typer.testing import CliRunner

from cli.main import app


class TestInitCommand:
    """Tests for init command."""

    def test_init_help(self, runner: CliRunner):
        """Test init --help."""
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "--template" in result.output
        assert "--display-name" in result.output
        assert "--force" in result.output

    def test_init_list_templates(self, runner: CliRunner):
        """Test init --list-templates."""
        result = runner.invoke(app, ["init", "--list-templates"])
        assert result.exit_code == 0
        assert "basic" in result.output.lower()
        assert "robot" in result.output.lower()
        assert "mode-aware" in result.output.lower()

    def test_init_basic_template(self, runner: CliRunner, mock_config_dir: Path):
        """Test init with basic template."""
        result = runner.invoke(app, ["init", "test_new_config"])
        assert result.exit_code == 0
        assert "created" in result.output.lower()

    def test_init_robot_template(self, runner: CliRunner, mock_config_dir: Path):
        """Test init with robot template."""
        result = runner.invoke(app, ["init", "test_robot", "--template", "robot"])
        assert result.exit_code == 0
        assert "created" in result.output.lower()

    def test_init_mode_aware_template(self, runner: CliRunner, mock_config_dir: Path):
        """Test init with mode-aware template."""
        result = runner.invoke(
            app, ["init", "test_new_modes", "--template", "mode-aware"]
        )
        assert result.exit_code == 0
        assert "created" in result.output.lower()

    def test_init_custom_display_name(self, runner: CliRunner, mock_config_dir: Path):
        """Test init with custom display name."""
        result = runner.invoke(
            app, ["init", "test_custom", "--display-name", "My Custom Bot"]
        )
        assert result.exit_code == 0

    def test_init_existing_without_force(
        self, runner: CliRunner, mock_config_dir: Path
    ):
        """Test init fails on existing config without --force."""
        runner.invoke(app, ["init", "test_existing"])
        result = runner.invoke(app, ["init", "test_existing"])
        assert result.exit_code == 1
        assert (
            "already exists" in result.output.lower()
            or "force" in result.output.lower()
        )

    def test_init_existing_with_force(self, runner: CliRunner, mock_config_dir: Path):
        """Test init overwrites with --force."""
        runner.invoke(app, ["init", "test_force"])
        result = runner.invoke(app, ["init", "test_force", "--force"])
        assert result.exit_code == 0

    def test_init_unknown_template(self, runner: CliRunner):
        """Test init with unknown template."""
        result = runner.invoke(app, ["init", "test_unknown", "--template", "unknown"])
        assert result.exit_code == 1
        assert "unknown" in result.output.lower()

    def test_init_no_name(self, runner: CliRunner):
        """Test init without name argument."""
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "provide" in result.output.lower() or "name" in result.output.lower()
