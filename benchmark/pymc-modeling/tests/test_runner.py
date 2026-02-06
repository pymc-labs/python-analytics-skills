"""Tests for the benchmark runner's skill injection mechanism.

These tests verify that:
1. with_skill condition actually injects SKILL.md into the Claude CLI command
2. no_skill condition does NOT inject any skill content
3. The skill content is the real SKILL.md file (not empty/corrupt)
4. The CLI command is constructed correctly for each condition
5. Result metadata records whether skill was injected

Run with: pixi run pytest tests/test_runner.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.runner import (
    SKILL_FILE,
    SKILL_MIN_CHARS,
    check_skill_status,
    load_skill_content,
    load_tasks,
    run_claude,
    run_task,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def skill_content():
    """Load the actual SKILL.md content."""
    assert SKILL_FILE.exists(), f"Skill not installed at {SKILL_FILE}"
    return SKILL_FILE.read_text()


@pytest.fixture
def fake_claude_success():
    """A mock subprocess result mimicking successful claude --print output."""
    output = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "```python\nimport pymc as pm\nprint('hello')\n```",
        "usage": {
            "input_tokens": 50,
            "cache_creation_input_tokens": 20000,
            "cache_read_input_tokens": 35000,
            "output_tokens": 500,
        },
        "total_cost_usd": 0.05,
        "num_turns": 1,
        "session_id": "test-session-123",
    }
    mock = MagicMock()
    mock.stdout = json.dumps(output)
    mock.stderr = ""
    mock.returncode = 0
    return mock


# ---------------------------------------------------------------------------
# Skill content integrity
# ---------------------------------------------------------------------------


class TestSkillContent:
    """Verify the skill file is present and intact."""

    def test_skill_file_exists(self):
        assert SKILL_FILE.exists(), (
            f"SKILL.md not found at {SKILL_FILE}. "
            "Install the pymc-modeling skill before running benchmarks."
        )

    def test_skill_file_minimum_size(self, skill_content):
        assert len(skill_content) >= SKILL_MIN_CHARS, (
            f"SKILL.md is only {len(skill_content)} chars, "
            f"expected >= {SKILL_MIN_CHARS}. File may be corrupted."
        )

    def test_skill_content_has_pymc_patterns(self, skill_content):
        """Verify the skill contains key PyMC guidance patterns."""
        assert "pm.Model" in skill_content
        assert "coords" in skill_content
        assert "dims" in skill_content
        assert "nutpie" in skill_content
        assert "az.summary" in skill_content or "arviz" in skill_content.lower()

    def test_load_skill_content_returns_full_file(self, skill_content):
        loaded = load_skill_content()
        assert loaded == skill_content

    def test_check_skill_status(self):
        status = check_skill_status()
        assert status["skill_installed"] is True
        assert status["skill_content_chars"] >= SKILL_MIN_CHARS


# ---------------------------------------------------------------------------
# CLI command construction
# ---------------------------------------------------------------------------


class TestCommandConstruction:
    """Verify that the CLI command is built correctly per condition."""

    @patch("src.runner.subprocess.run")
    def test_with_skill_appends_system_prompt(self, mock_run, fake_claude_success):
        """with_skill MUST pass --append-system-prompt with SKILL.md content."""
        mock_run.return_value = fake_claude_success

        run_claude("test prompt", condition="with_skill", model="sonnet")

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        # --append-system-prompt must be in the command
        assert "--append-system-prompt" in cmd, (
            "with_skill condition did not include --append-system-prompt in CLI command"
        )

        # The argument after --append-system-prompt must be the skill content
        idx = cmd.index("--append-system-prompt")
        injected = cmd[idx + 1]
        real_skill = SKILL_FILE.read_text()
        assert injected == real_skill, (
            "The content passed to --append-system-prompt does not match SKILL.md"
        )

    @patch("src.runner.subprocess.run")
    def test_no_skill_omits_system_prompt(self, mock_run, fake_claude_success):
        """no_skill MUST NOT pass --append-system-prompt."""
        mock_run.return_value = fake_claude_success

        run_claude("test prompt", condition="no_skill", model="sonnet")

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        assert "--append-system-prompt" not in cmd, (
            "no_skill condition should NOT include --append-system-prompt"
        )

    @patch("src.runner.subprocess.run")
    def test_with_skill_cmd_has_correct_base_flags(self, mock_run, fake_claude_success):
        """Both conditions should have standard CLI flags."""
        mock_run.return_value = fake_claude_success

        run_claude("test prompt", condition="with_skill", model="sonnet")

        cmd = mock_run.call_args[0][0]
        assert "claude" in cmd[0]
        assert "--print" in cmd
        assert "--output-format" in cmd
        assert "--no-session-persistence" in cmd


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------


class TestResultMetadata:
    """Verify that results record skill injection state."""

    @patch("src.runner.subprocess.run")
    def test_with_skill_result_records_injection(self, mock_run, fake_claude_success):
        mock_run.return_value = fake_claude_success

        result = run_claude("test prompt", condition="with_skill")

        assert result["skill_injected"] is True
        assert result["skill_chars_injected"] >= SKILL_MIN_CHARS
        assert result["condition"] == "with_skill"

    @patch("src.runner.subprocess.run")
    def test_no_skill_result_records_no_injection(self, mock_run, fake_claude_success):
        mock_run.return_value = fake_claude_success

        result = run_claude("test prompt", condition="no_skill")

        assert result["skill_injected"] is False
        assert result["skill_chars_injected"] == 0
        assert result["condition"] == "no_skill"

    @patch("src.runner.subprocess.run")
    def test_result_has_total_input_tokens(self, mock_run, fake_claude_success):
        """Results must include the full token count (including cache)."""
        mock_run.return_value = fake_claude_success

        result = run_claude("test prompt", condition="no_skill")

        assert result["total_input_tokens"] is not None
        assert result["total_input_tokens"] > 0
        # 50 + 20000 + 35000 = 55050
        assert result["total_input_tokens"] == 55050


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


class TestGuardRails:
    """Verify that the runner fails fast on bad state."""

    def test_load_skill_content_fails_if_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.runner.SKILL_FILE", tmp_path / "nonexistent.md")
        with pytest.raises(FileNotFoundError, match="Skill file not found"):
            load_skill_content()

    def test_load_skill_content_fails_if_too_small(self, tmp_path, monkeypatch):
        tiny = tmp_path / "SKILL.md"
        tiny.write_text("tiny")
        monkeypatch.setattr("src.runner.SKILL_FILE", tiny)
        with pytest.raises(ValueError, match="suspiciously small"):
            load_skill_content()

    def test_run_task_rejects_invalid_condition(self):
        with pytest.raises(ValueError, match="Unknown condition"):
            run_task("T1", "maybe_skill")

    @patch("src.runner.subprocess.run")
    def test_with_skill_preflight_check(self, mock_run, tmp_path, monkeypatch):
        """run_task should fail before calling Claude if skill is missing."""
        monkeypatch.setattr("src.runner.SKILL_FILE", tmp_path / "nonexistent.md")
        with pytest.raises(FileNotFoundError):
            run_task("T1", "with_skill", force=True)
        # Claude should never have been called
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: dry-run shows skill info
# ---------------------------------------------------------------------------


class TestDryRun:
    """Verify dry-run mode reports skill injection details."""

    def test_dry_run_with_skill_returns_metadata(self):
        result = run_task("T1", "with_skill", dry_run=True)
        assert result is not None
        assert result["condition"] == "with_skill"

    def test_dry_run_no_skill_returns_metadata(self):
        result = run_task("T1", "no_skill", dry_run=True)
        assert result is not None
        assert result["condition"] == "no_skill"
