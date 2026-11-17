"""Compatibility CLI for generic MobileNetV3 embedding/classification training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.embedding_workflow import *  # noqa: F401,F403,E402
from src.training.embedding_workflow import main  # noqa: E402


if __name__ == "__main__":
    main()
