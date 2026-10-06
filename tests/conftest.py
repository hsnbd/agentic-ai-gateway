"""Settings shared by every test suite."""

from __future__ import annotations

import os
from pathlib import Path

# Keep tiktoken's downloaded BPE files across runs and reboots (the default is
# the OS temp dir), so token counting never waits on the network mid-suite.
os.environ.setdefault(
    "TIKTOKEN_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".cache" / "tiktoken")
)
