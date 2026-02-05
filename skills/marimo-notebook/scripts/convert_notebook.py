#!/usr/bin/env python3
"""Convert Jupyter notebooks to marimo format."""

import subprocess
import sys
from pathlib import Path


def convert_jupyter_to_marimo(input_path: str, output_path: str | None = None) -> str:
    """Convert a Jupyter notebook to marimo format.

    Args:
        input_path: Path to .ipynb file (local or GitHub URL)
        output_path: Optional output path. If None, derives from input.

    Returns:
        Path to the created marimo notebook.
    """
    input_file = Path(input_path)

    if output_path is None:
        output_path = str(input_file.with_suffix(".py"))

    cmd = ["marimo", "convert", input_path, "-o", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed: {result.stderr}")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: convert_notebook.py <input.ipynb> [output.py]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = convert_jupyter_to_marimo(input_file, output_file)
        print(f"Converted to: {result}")
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
