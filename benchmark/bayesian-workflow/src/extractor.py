"""Workflow trace extraction from turns.jsonl.

Parses the turn-by-turn transcript to identify workflow steps:
- Model versions (Write to model*.py)
- Model fits (Bash running python model*.py)
- Prior predictive checks
- Posterior predictive checks
- LOO/model comparisons
- Convergence diagnostics
- Reasoning excerpts
"""

import json
import re
from pathlib import Path

# Patterns for detecting workflow steps in Bash commands AND written Python code
PRIOR_PREDICTIVE_PATTERNS = [
    r"sample_prior_predictive",
    r"prior_predictive",
    r"plot_prior",
]

POSTERIOR_PREDICTIVE_PATTERNS = [
    r"sample_posterior_predictive",
    r"posterior_predictive",
    r"plot_ppc",
    r"ppc",
]

LOO_COMPARISON_PATTERNS = [
    r"az\.loo",
    r"az\.compare",
    r"az\.waic",
    r"loo\(",
    r"compare\(",
]

DIAGNOSTICS_PATTERNS = [
    r"az\.summary",
    r"az\.plot_trace",
    r"az\.plot_forest",
    r"az\.ess",
    r"az\.rhat",
    r"r_hat",
    r"ess_bulk",
    r"ess_tail",
    r"divergen",
]

# Keywords that indicate reasoning about workflow
REASONING_KEYWORDS = [
    "start with", "begin with", "simple model", "baseline",
    "expand", "add complexity", "compare", "iterate",
    "prior predictive", "posterior predictive",
    "convergence", "diagnostic", "divergen",
    "loo", "model comparison", "waic",
    "improve", "revise", "next model", "previous model",
]


def extract_workflow_trace(run_dir: Path) -> dict:
    """Parse turns.jsonl into a structured workflow trace.

    Returns a dict with:
        models: list of model version dicts (file, content_hash, write_index)
        total_models_written: int
        total_models_fitted: int
        prior_predictive_count: int
        posterior_predictive_count: int
        loo_comparisons: int
        diagnostics_count: int
        reasoning_excerpts: list of str
        tool_sequence: list of (tool_name, summary) tuples
    """
    turns_path = run_dir / "turns.jsonl"

    trace = {
        "models": [],
        "total_models_written": 0,
        "total_models_fitted": 0,
        "prior_predictive_count": 0,
        "posterior_predictive_count": 0,
        "loo_comparisons": 0,
        "diagnostics_count": 0,
        "reasoning_excerpts": [],
        "tool_sequence": [],
    }

    if not turns_path.exists():
        return trace

    turns = _load_turns(turns_path)
    model_contents = {}  # track model file contents for diffing
    write_index = 0

    for turn in turns:
        content = turn.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            block_type = block.get("type")

            if block_type == "text":
                text = block.get("text", "")
                _extract_reasoning(text, trace)

            elif block_type == "tool_use":
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})

                if tool_name == "Write":
                    _process_write(tool_input, trace, model_contents, write_index)
                    write_index += 1

                elif tool_name == "Bash":
                    _process_bash(tool_input, trace)

                trace["tool_sequence"].append(
                    (tool_name, _summarize_tool_call(tool_name, tool_input))
                )

    return trace


def _load_turns(turns_path: Path) -> list[dict]:
    """Load turns from JSONL file."""
    turns = []
    with open(turns_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                turns.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return turns


def _process_write(tool_input: dict, trace: dict, model_contents: dict, idx: int):
    """Process a Write tool call — detect model file writes and scan code content."""
    file_path = tool_input.get("file_path", "")
    content = tool_input.get("content", "")

    # Track model file writes
    if re.search(r"model.*\.py$", file_path):
        trace["total_models_written"] += 1

        model_info = {
            "file": file_path,
            "write_index": idx,
            "content_length": len(content),
            "is_new_version": file_path not in model_contents,
        }

        if file_path in model_contents:
            model_info["is_rewrite"] = True
        else:
            model_info["is_rewrite"] = False

        model_contents[file_path] = content
        trace["models"].append(model_info)

    # Scan ALL .py file content for workflow patterns
    # (Claude puts prior/posterior predictive calls inside model scripts,
    #  not in standalone bash commands)
    if file_path.endswith(".py") and content:
        _scan_code_content(content, trace)


def _scan_code_content(content: str, trace: dict):
    """Scan written Python code for workflow patterns.

    This catches prior/posterior predictive checks, LOO comparisons, and
    diagnostics that are embedded in model scripts rather than run as
    standalone bash commands.
    """
    if any(re.search(p, content) for p in PRIOR_PREDICTIVE_PATTERNS):
        trace["prior_predictive_count"] += 1

    if any(re.search(p, content) for p in POSTERIOR_PREDICTIVE_PATTERNS):
        trace["posterior_predictive_count"] += 1

    if any(re.search(p, content) for p in LOO_COMPARISON_PATTERNS):
        trace["loo_comparisons"] += 1

    if any(re.search(p, content) for p in DIAGNOSTICS_PATTERNS):
        trace["diagnostics_count"] += 1


def _process_bash(tool_input: dict, trace: dict):
    """Process a Bash tool call — detect model fits, checks, comparisons."""
    command = tool_input.get("command", "")

    # Model execution
    if re.search(r"python\s+.*model.*\.py", command):
        trace["total_models_fitted"] += 1

    # Prior predictive checks
    if any(re.search(p, command) for p in PRIOR_PREDICTIVE_PATTERNS):
        trace["prior_predictive_count"] += 1

    # Posterior predictive checks
    if any(re.search(p, command) for p in POSTERIOR_PREDICTIVE_PATTERNS):
        trace["posterior_predictive_count"] += 1

    # LOO / model comparisons
    if any(re.search(p, command) for p in LOO_COMPARISON_PATTERNS):
        trace["loo_comparisons"] += 1

    # Convergence diagnostics
    if any(re.search(p, command) for p in DIAGNOSTICS_PATTERNS):
        trace["diagnostics_count"] += 1


def _extract_reasoning(text: str, trace: dict):
    """Extract reasoning excerpts from text blocks."""
    text_lower = text.lower()
    for keyword in REASONING_KEYWORDS:
        if keyword in text_lower:
            # Extract the sentence containing the keyword
            for sentence in re.split(r'[.!?\n]', text):
                if keyword in sentence.lower() and len(sentence.strip()) > 20:
                    excerpt = sentence.strip()[:200]
                    if excerpt not in trace["reasoning_excerpts"]:
                        trace["reasoning_excerpts"].append(excerpt)
            break  # one keyword match per text block is enough


def _summarize_tool_call(tool_name: str, tool_input: dict) -> str:
    """Create a short summary of a tool call."""
    if tool_name == "Write":
        return f"write {tool_input.get('file_path', '?')}"
    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")[:80]
        return f"bash: {cmd}"
    elif tool_name == "Read":
        return f"read {tool_input.get('file_path', '?')}"
    else:
        return tool_name
