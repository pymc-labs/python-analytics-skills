"""Tests for the benchmark runner."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.runner import (
    TASKS_PATH,
    _build_command,
    _build_prompt,
    _parse_response,
    get_run_dir,
    is_cached,
    is_corrupted_model,
    load_tasks,
    verify_isolation,
    verify_token_difference,
)


def test_load_tasks():
    """Tasks load correctly from YAML."""
    config = load_tasks()
    assert "preamble" in config
    assert "tasks" in config
    assert len(config["tasks"]) == 5
    assert "T1_hierarchical" in config["tasks"]
    assert "T2_ordinal" in config["tasks"]


def test_task_has_required_fields():
    """Each task has prompt, judge_rubric, best_practices_patterns."""
    config = load_tasks()
    for tid, task in config["tasks"].items():
        assert "prompt" in task, f"{tid} missing prompt"
        assert "judge_rubric" in task, f"{tid} missing judge_rubric"
        assert "best_practices_patterns" in task, f"{tid} missing best_practices_patterns"
        assert "name" in task, f"{tid} missing name"
        assert "data_files" in task, f"{tid} missing data_files"


def test_build_prompt():
    """Prompt combines preamble and task prompt."""
    result = _build_prompt("Preamble text.", "Task prompt.")
    assert "Preamble text." in result
    assert "Task prompt." in result


def test_build_command_no_skill():
    """no_skill command has no --append-system-prompt."""
    cmd = _build_command("no_skill")
    assert "--append-system-prompt" not in cmd
    assert "--print" in cmd
    assert "--model" in cmd


def test_build_command_with_skill():
    """with_skill command includes --append-system-prompt."""
    cmd = _build_command("with_skill")
    assert "--append-system-prompt" in cmd
    idx = cmd.index("--append-system-prompt")
    skill_content = cmd[idx + 1]
    assert len(skill_content) > 1000  # SKILL.md is ~18KB


def test_get_run_dir():
    """Run dir follows naming convention."""
    d = get_run_dir("T1_hierarchical", "no_skill", 2)
    assert d.name == "T1_hierarchical_no_skill_rep2"


def test_parse_response_valid():
    """Parse a valid stream-json NDJSON response."""
    lines = [
        json.dumps({"type": "system", "subtype": "init", "data": {}}),
        json.dumps({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Write", "input": {"file_path": "model.py"}},
                ]
            },
        }),
        json.dumps({
            "type": "result",
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 200,
                "cache_read_input_tokens": 300,
                "output_tokens": 400,
            },
            "result": "Created model.py and results.nc",
            "num_turns": 5,
            "is_error": False,
            "total_cost_usd": 0.15,
            "permission_denials": [],
        }),
    ]
    response = "\n".join(lines)
    parsed = _parse_response(response)
    assert parsed["input_tokens"] == 100
    assert parsed["cache_creation_tokens"] == 200
    assert parsed["cache_read_tokens"] == 300
    assert parsed["total_input_tokens"] == 600
    assert parsed["output_tokens"] == 400
    assert parsed["num_turns"] == 5
    assert parsed["tool_calls"] == []
    assert len(parsed["turns"]) == 1


def test_parse_response_with_denials():
    """Permission denials are captured as tool calls."""
    lines = [
        json.dumps({
            "type": "result",
            "usage": {"input_tokens": 100, "output_tokens": 200},
            "result": "",
            "num_turns": 3,
            "permission_denials": [
                {"tool_name": "Write", "tool_use_id": "x"},
                {"tool_name": "Bash", "tool_use_id": "y"},
            ],
        }),
    ]
    response = "\n".join(lines)
    parsed = _parse_response(response)
    assert parsed["tool_calls"] == ["Write", "Bash"]


def test_parse_response_invalid():
    """Parse invalid NDJSON returns error."""
    parsed = _parse_response("not json at all")
    assert "error" in parsed


def test_parse_response_extracts_turns():
    """Verify turns list is populated from assistant messages."""
    lines = [
        json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Let me write model.py"}]},
        }),
        json.dumps({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Write", "input": {"file_path": "model.py"}},
                ]
            },
        }),
        json.dumps({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Bash", "input": {"command": "python model.py"}},
                ]
            },
        }),
        json.dumps({
            "type": "result",
            "usage": {"input_tokens": 50, "output_tokens": 100},
            "result": "Done",
            "num_turns": 3,
            "is_error": False,
            "total_cost_usd": 0.05,
            "permission_denials": [],
        }),
    ]
    response = "\n".join(lines)
    parsed = _parse_response(response)
    assert len(parsed["turns"]) == 3
    assert parsed["turns"][0]["message"]["content"][0]["type"] == "text"
    assert parsed["turns"][1]["message"]["content"][0]["name"] == "Write"


def test_verify_isolation_clean():
    """Clean response passes isolation check."""
    parsed = {"tool_calls": ["Bash", "Write", "Read"]}
    failures = verify_isolation(parsed, "no_skill")
    assert failures == []


def test_verify_isolation_skill_tool():
    """Skill tool call triggers isolation failure."""
    parsed = {"tool_calls": ["Bash", "Skill", "Write"]}
    failures = verify_isolation(parsed, "no_skill")
    assert len(failures) == 1
    assert "Skill" in failures[0]


def test_verify_token_difference_ok():
    """Both runs completed — passes."""
    ns = {"num_turns": 5, "cache_creation_tokens": 5000}
    ws = {"num_turns": 6, "cache_creation_tokens": 9500}
    failures = verify_token_difference(ns, ws)
    assert failures == []


def test_verify_token_difference_no_turns():
    """Zero turns triggers failure."""
    ns = {"num_turns": 0, "cache_creation_tokens": 0}
    ws = {"num_turns": 5, "cache_creation_tokens": 9500}
    failures = verify_token_difference(ns, ws)
    assert len(failures) == 1
    assert "no_skill" in failures[0]


def test_is_cached_no_dir(tmp_path):
    """Non-existent run dir is not cached."""
    with patch("src.runner.RUNS_DIR", tmp_path):
        assert not is_cached("T1_hierarchical", "no_skill", 0)


def test_is_cached_with_metadata(tmp_path):
    """Run dir with metadata.json is cached."""
    run_dir = tmp_path / "T1_hierarchical_no_skill_rep0"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text("{}")
    with patch("src.runner.RUNS_DIR", tmp_path):
        assert is_cached("T1_hierarchical", "no_skill", 0)


class TestIsCorruptedModel:
    """Tests for is_corrupted_model corruption detection."""

    def test_library_file_with_copyright(self, tmp_path):
        """File with Copyright header is detected as corrupted."""
        model = tmp_path / "model.py"
        model.write_text(
            '#   Copyright 2024 The PyMC Developers\n'
            '#   Licensed under the Apache License, Version 2.0\n'
            'import pymc as pm\n'
        )
        assert is_corrupted_model(model)

    def test_library_file_with_licensed_under(self, tmp_path):
        """File with Licensed under header is detected as corrupted."""
        model = tmp_path / "model.py"
        model.write_text(
            '# Licensed under the MIT License\n'
            'def foo(): pass\n'
        )
        assert is_corrupted_model(model)

    def test_normal_model_not_corrupted(self, tmp_path):
        """Normal Claude-generated model.py is not flagged."""
        model = tmp_path / "model.py"
        model.write_text(
            'import pymc as pm\n'
            'import numpy as np\n'
            'with pm.Model() as model:\n'
            '    mu = pm.Normal("mu", 0, 1)\n'
        )
        assert not is_corrupted_model(model)

    def test_nonexistent_file(self, tmp_path):
        """Non-existent file returns False (not corrupted)."""
        model = tmp_path / "model.py"
        assert not is_corrupted_model(model)
