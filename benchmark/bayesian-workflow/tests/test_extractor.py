"""Tests for the workflow trace extractor."""

import json
from pathlib import Path

import pytest

from src.extractor import extract_workflow_trace


def _make_assistant_turn(tool_name, tool_input):
    """Helper: create an assistant turn with a single tool_use block."""
    return {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "name": tool_name, "input": tool_input},
            ]
        },
    }


def _make_text_turn(text):
    """Helper: create an assistant turn with a text block."""
    return {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": text},
            ]
        },
    }


def _write_turns(run_dir: Path, turns: list[dict]):
    """Helper: write turns.jsonl."""
    path = run_dir / "turns.jsonl"
    with open(path, "w") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")


class TestExtractWorkflowTrace:
    def test_empty_run_dir(self, tmp_path):
        """No turns.jsonl → empty trace."""
        trace = extract_workflow_trace(tmp_path)
        assert trace["total_models_written"] == 0
        assert trace["total_models_fitted"] == 0

    def test_single_model_write(self, tmp_path):
        """Single Write to model.py detected."""
        turns = [
            _make_assistant_turn("Write", {
                "file_path": "/tmp/work/model.py",
                "content": "import pymc as pm\n",
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["total_models_written"] == 1
        assert len(trace["models"]) == 1
        assert trace["models"][0]["is_rewrite"] is False

    def test_model_rewrite_detected(self, tmp_path):
        """Second Write to same file is a rewrite."""
        turns = [
            _make_assistant_turn("Write", {
                "file_path": "/tmp/work/model.py",
                "content": "import pymc as pm\n# v1",
            }),
            _make_assistant_turn("Write", {
                "file_path": "/tmp/work/model.py",
                "content": "import pymc as pm\n# v2",
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["total_models_written"] == 2
        assert trace["models"][0]["is_rewrite"] is False
        assert trace["models"][1]["is_rewrite"] is True

    def test_multiple_model_files(self, tmp_path):
        """Writes to different model files are separate versions."""
        turns = [
            _make_assistant_turn("Write", {
                "file_path": "/tmp/work/model.py",
                "content": "# model v1",
            }),
            _make_assistant_turn("Write", {
                "file_path": "/tmp/work/model_v2.py",
                "content": "# model v2",
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["total_models_written"] == 2
        assert trace["models"][0]["is_new_version"] is True
        assert trace["models"][1]["is_new_version"] is True

    def test_model_fit_detected(self, tmp_path):
        """Bash running python model.py → model fit."""
        turns = [
            _make_assistant_turn("Bash", {"command": "python model.py"}),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["total_models_fitted"] == 1

    def test_model_fit_with_path(self, tmp_path):
        """Bash running python /path/to/model_v2.py → model fit."""
        turns = [
            _make_assistant_turn("Bash", {"command": "python /tmp/work/model_v2.py"}),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["total_models_fitted"] == 1

    def test_prior_predictive_detected(self, tmp_path):
        """Bash with sample_prior_predictive → detected."""
        turns = [
            _make_assistant_turn("Bash", {
                "command": "python -c 'import pymc as pm; pm.sample_prior_predictive()'"
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["prior_predictive_count"] == 1

    def test_posterior_predictive_detected(self, tmp_path):
        """Bash with sample_posterior_predictive → detected."""
        turns = [
            _make_assistant_turn("Bash", {
                "command": "python -c 'pm.sample_posterior_predictive(idata)'"
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["posterior_predictive_count"] == 1

    def test_ppc_plot_detected(self, tmp_path):
        """Bash with plot_ppc → posterior predictive detected."""
        turns = [
            _make_assistant_turn("Bash", {
                "command": "python -c 'az.plot_ppc(idata)'"
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["posterior_predictive_count"] == 1

    def test_loo_detected(self, tmp_path):
        """Bash with az.loo → LOO comparison detected."""
        turns = [
            _make_assistant_turn("Bash", {
                "command": "python -c 'az.loo(idata)'"
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["loo_comparisons"] == 1

    def test_compare_detected(self, tmp_path):
        """Bash with az.compare → LOO comparison detected."""
        turns = [
            _make_assistant_turn("Bash", {
                "command": "python -c 'az.compare({\"m1\": idata1, \"m2\": idata2})'"
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["loo_comparisons"] == 1

    def test_diagnostics_detected(self, tmp_path):
        """Bash with az.summary → diagnostics detected."""
        turns = [
            _make_assistant_turn("Bash", {
                "command": "python -c 'az.summary(idata)'"
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["diagnostics_count"] == 1

    def test_reasoning_extraction(self, tmp_path):
        """Text blocks with workflow keywords → reasoning excerpts."""
        turns = [
            _make_text_turn(
                "Let me start with a simple model as a baseline. "
                "I'll expand the model if diagnostics suggest problems."
            ),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert len(trace["reasoning_excerpts"]) >= 1

    def test_non_model_write_ignored(self, tmp_path):
        """Write to non-model file is not counted."""
        turns = [
            _make_assistant_turn("Write", {
                "file_path": "/tmp/work/analysis.py",
                "content": "import polars\n",
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["total_models_written"] == 0

    def test_full_workflow_trace(self, tmp_path):
        """Integration test: full workflow with multiple steps."""
        turns = [
            _make_text_turn("Let me start with a simple model."),
            _make_assistant_turn("Write", {
                "file_path": "/tmp/work/model.py",
                "content": "# simple model v1",
            }),
            _make_assistant_turn("Bash", {"command": "python model.py"}),
            _make_assistant_turn("Bash", {
                "command": "python -c 'az.summary(idata)'"
            }),
            _make_text_turn("The simple model converged. Let me expand with a hierarchical structure."),
            _make_assistant_turn("Write", {
                "file_path": "/tmp/work/model_v2.py",
                "content": "# hierarchical model v2",
            }),
            _make_assistant_turn("Bash", {"command": "python model_v2.py"}),
            _make_assistant_turn("Bash", {
                "command": "python -c 'az.compare({\"m1\": idata1, \"m2\": idata2})'"
            }),
            _make_assistant_turn("Bash", {
                "command": "python -c 'az.plot_ppc(idata)'"
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["total_models_written"] == 2
        assert trace["total_models_fitted"] == 2
        assert trace["loo_comparisons"] == 1
        assert trace["posterior_predictive_count"] == 1
        assert trace["diagnostics_count"] == 1
        assert len(trace["reasoning_excerpts"]) >= 1
        assert len(trace["tool_sequence"]) == 7  # 2 writes + 2 bash model + 3 bash checks

    def test_prior_predictive_in_written_code(self, tmp_path):
        """Prior predictive call inside model.py content is detected."""
        turns = [
            _make_assistant_turn("Write", {
                "file_path": "/tmp/work/model.py",
                "content": (
                    "import pymc as pm\n"
                    "with model:\n"
                    "    prior = pm.sample_prior_predictive(draws=500)\n"
                    "    idata = pm.sample()\n"
                ),
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["prior_predictive_count"] == 1

    def test_posterior_predictive_in_written_code(self, tmp_path):
        """Posterior predictive call inside model.py content is detected."""
        turns = [
            _make_assistant_turn("Write", {
                "file_path": "/tmp/work/model.py",
                "content": (
                    "import pymc as pm\n"
                    "with model:\n"
                    "    idata = pm.sample()\n"
                    "    pm.sample_posterior_predictive(idata, extend_inferencedata=True)\n"
                ),
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["posterior_predictive_count"] == 1

    def test_loo_in_written_code(self, tmp_path):
        """LOO call inside model.py content is detected."""
        turns = [
            _make_assistant_turn("Write", {
                "file_path": "/tmp/work/model.py",
                "content": (
                    "import arviz as az\n"
                    "comparison = az.compare({'m1': idata1, 'm2': idata2})\n"
                ),
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["loo_comparisons"] == 1

    def test_diagnostics_in_written_code(self, tmp_path):
        """Diagnostics call inside model.py content is detected."""
        turns = [
            _make_assistant_turn("Write", {
                "file_path": "/tmp/work/model.py",
                "content": (
                    "import arviz as az\n"
                    "print(az.summary(idata))\n"
                ),
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        assert trace["diagnostics_count"] == 1

    def test_non_model_py_content_scanned(self, tmp_path):
        """Workflow patterns in analysis.py (non-model file) are also detected."""
        turns = [
            _make_assistant_turn("Write", {
                "file_path": "/tmp/work/analysis.py",
                "content": (
                    "import arviz as az\n"
                    "az.plot_ppc(idata)\n"
                    "az.loo(idata)\n"
                ),
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        # analysis.py is not a model file, so no model written
        assert trace["total_models_written"] == 0
        # But workflow patterns should be detected in the code content
        assert trace["posterior_predictive_count"] == 1
        assert trace["loo_comparisons"] == 1

    def test_both_bash_and_write_counted(self, tmp_path):
        """Patterns found in both Bash and Write are counted separately."""
        turns = [
            _make_assistant_turn("Write", {
                "file_path": "/tmp/work/model.py",
                "content": "pm.sample_prior_predictive(draws=500)\n",
            }),
            _make_assistant_turn("Bash", {
                "command": "python -c 'pm.sample_prior_predictive()'"
            }),
        ]
        _write_turns(tmp_path, turns)
        trace = extract_workflow_trace(tmp_path)
        # Both the written code AND the bash command contain prior predictive
        assert trace["prior_predictive_count"] == 2

    def test_malformed_jsonl(self, tmp_path):
        """Malformed lines are skipped gracefully."""
        path = tmp_path / "turns.jsonl"
        path.write_text("not valid json\n{}\n")
        trace = extract_workflow_trace(tmp_path)
        assert trace["total_models_written"] == 0
