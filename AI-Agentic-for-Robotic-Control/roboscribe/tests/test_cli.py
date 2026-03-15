"""Tests for CLI commands."""

from click.testing import CliRunner

from roboscribe.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "RoboScribe" in result.output
    assert "generate" in result.output
    assert "test" in result.output
    assert "envs" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_envs_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["envs"])
    assert result.exit_code == 0
    assert "Lift" in result.output
    assert "Stack" in result.output
    assert "Door" in result.output
    assert "Wipe" in result.output


def test_generate_no_api_key():
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "pick up cube", "--env", "Lift"])
    assert result.exit_code != 0
    assert "API key" in result.output or "Config error" in result.output


def test_test_missing_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["test", "nonexistent.py"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "File not found" in result.output
