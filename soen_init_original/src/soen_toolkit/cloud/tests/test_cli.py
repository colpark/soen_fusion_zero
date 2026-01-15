"""Tests for CLI functionality."""

from click.testing import CliRunner
import pytest

from soen_toolkit.cloud.cli import cli


class TestCLI:
    """Test CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_help(self, runner):
        """Test that --help works."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "SOEN Cloud" in result.output
        assert "train" in result.output
        assert "status" in result.output

    def test_train_help(self, runner):
        """Test train command help."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--estimate" in result.output
        assert "--instance-type" in result.output

    def test_instances_command(self, runner):
        """Test instances command."""
        result = runner.invoke(cli, ["instances"])
        assert result.exit_code == 0
        assert "ml.g5.xlarge" in result.output
        assert "On-Demand" in result.output
        assert "Spot" in result.output

    def test_status_help(self, runner):
        """Test status command help."""
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0
        assert "JOB_NAME" in result.output

    def test_list_help(self, runner):
        """Test list command help."""
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0
        assert "--project" in result.output
        assert "--limit" in result.output

    def test_stop_help(self, runner):
        """Test stop command help."""
        result = runner.invoke(cli, ["stop", "--help"])
        assert result.exit_code == 0
        assert "JOB_NAME" in result.output

    def test_train_missing_config(self, runner):
        """Test train command requires config file."""
        result = runner.invoke(cli, ["train"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_train_nonexistent_config(self, runner):
        """Test train command with nonexistent config file."""
        result = runner.invoke(cli, ["train", "--config", "/nonexistent/config.yaml"])
        assert result.exit_code != 0

