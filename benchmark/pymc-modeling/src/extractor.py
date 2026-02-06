"""Extract Python scripts from Claude CLI output.

Parses response text for fenced code blocks and writes them as
standalone .py scripts ready for execution.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BENCHMARK_DIR / "results" / "raw"
CODE_DIR = BENCHMARK_DIR / "results" / "code"


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from markdown-formatted text."""
    blocks = []

    # Match ```python ... ``` blocks
    python_blocks = re.findall(
        r"```python\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE
    )
    blocks.extend(python_blocks)

    if blocks:
        return blocks

    # Fallback: match generic ``` ... ``` blocks that look like Python
    generic_blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    for block in generic_blocks:
        if any(kw in block for kw in ("import", "def ", "pm.", "pymc")):
            blocks.append(block)

    return blocks


def convert_marimo_to_script(code: str) -> str:
    """Convert a marimo notebook to a sequential Python script."""
    # Extract cell bodies from @app.cell decorated functions
    cell_pattern = re.compile(
        r"@app\.cell.*?\ndef\s+\w+\([^)]*\):\s*\n(.*?)(?=\n@app\.cell|\nif __name__|$)",
        re.DOTALL,
    )
    cells = cell_pattern.findall(code)

    if not cells:
        return code

    # Collect imports and cell bodies
    lines = []
    for cell in cells:
        # Dedent cell body (remove one level of indentation)
        dedented = []
        for line in cell.split("\n"):
            if line.startswith("    "):
                dedented.append(line[4:])
            elif line.strip() == "":
                dedented.append("")
            else:
                dedented.append(line)
        # Skip return statements (marimo cell artifacts)
        body = "\n".join(
            l
            for l in dedented
            if not l.strip().startswith("return ")
            and not l.strip().startswith("return(")
        )
        lines.append(body.strip())

    return "\n\n".join(lines)


def extract_from_result(result: dict) -> dict:
    """Extract code from a single benchmark result.

    Returns dict with extraction status and code content.
    """
    extraction = {
        "task_id": result.get("task_id", ""),
        "condition": result.get("condition", ""),
        "replication": result.get("replication", 0),
        "success": False,
        "code": None,
        "error": None,
        "source": None,
        "is_marimo": False,
    }

    # Skip if the Claude run itself failed
    if not result.get("success"):
        extraction["error"] = f"Claude run failed: {result.get('error', 'unknown')}"
        return extraction

    response_text = result.get("response_text", "")
    if not response_text:
        extraction["error"] = "No response text"
        return extraction

    # Also check permission_denials for code (prior art approach)
    code_from_denials = []
    response_json_str = result.get("response", "")
    if response_json_str:
        try:
            rj = json.loads(response_json_str)
            for denial in rj.get("permission_denials", []):
                content = denial.get("tool_input", {}).get("content", "")
                if content and "import" in content:
                    code_from_denials.append(content)
        except (json.JSONDecodeError, TypeError):
            pass

    # Try permission denial code first (most complete)
    if code_from_denials:
        code = max(code_from_denials, key=len)
        extraction["source"] = "permission_denial"
    else:
        # Extract from markdown code blocks
        blocks = extract_code_blocks(response_text)
        if not blocks:
            extraction["error"] = "No code blocks found in response"
            return extraction
        code = "\n\n".join(blocks)
        extraction["source"] = "code_blocks"

    # Check for marimo notebook and convert
    if "marimo.App" in code or "@app.cell" in code:
        extraction["is_marimo"] = True
        code = convert_marimo_to_script(code)

    # Validate: must contain pymc import
    if "import pymc" not in code and "from pymc" not in code:
        extraction["error"] = "Extracted code does not import pymc"
        return extraction

    extraction["success"] = True
    extraction["code"] = code
    return extraction


def save_extracted_code(extraction: dict) -> Path | None:
    """Save extracted code to a .py file."""
    if not extraction["success"] or not extraction["code"]:
        return None

    CODE_DIR.mkdir(parents=True, exist_ok=True)
    task_id = extraction["task_id"]
    condition = extraction["condition"]
    rep = extraction["replication"]
    filename = f"{task_id}_{condition}_rep{rep}.py"
    path = CODE_DIR / filename
    path.write_text(extraction["code"])
    return path


def extract_all() -> list[dict]:
    """Extract code from all raw results."""
    if not RAW_DIR.exists():
        print("No raw results found.")
        return []

    results = []
    for path in sorted(RAW_DIR.glob("*.json")):
        with open(path) as f:
            result = json.load(f)

        extraction = extract_from_result(result)
        code_path = save_extracted_code(extraction)

        status = "OK" if extraction["success"] else f"FAIL: {extraction['error']}"
        print(f"  {path.stem}: {status}")
        if code_path:
            extraction["code_path"] = str(code_path)

        results.append(extraction)

    success = sum(1 for r in results if r["success"])
    print(f"\nExtracted {success}/{len(results)} scripts")
    return results
