"""Project-local resource paths, with optional environment overrides."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_DIR = Path(os.environ.get("WORLDMEMBENCH_DATASETS_DIR", PROJECT_ROOT)).expanduser()
DEFAULT_WEIGHTS_DIR = Path(
    os.environ.get("WORLDMEMBENCH_WEIGHTS_DIR", DEFAULT_DATASETS_DIR / "weights")
).expanduser()
