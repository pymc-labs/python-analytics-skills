"""Root conftest.py — ensures `src` is importable as a package."""

import sys
from pathlib import Path

# Add the benchmark directory to sys.path so `from src.runner import ...` works
sys.path.insert(0, str(Path(__file__).parent))
