"""Compatibility CLI for AXE train-gallery/validation-query evaluation.

The implementation lives in :mod:`src.evaluation.gallery` so notebooks and
other entry points use the same audited metrics and leakage checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.gallery import *  # noqa: F401,F403,E402
from src.evaluation.gallery import main  # noqa: E402


if __name__ == "__main__":
    main()
