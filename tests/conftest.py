from __future__ import annotations

import pytest

from celltrack_bench import render as R

DATA = R.REPO_ROOT / "data"


def requires_benchmark(name: str = "hard"):
    """Skip tests that need data/<name>, which is generated locally (construction/rebuild_benchmark.sh)."""
    return pytest.mark.skipif(
        not (DATA / name / "test_items.json").is_file(),
        reason=f"data/{name} not generated; run construction/rebuild_benchmark.sh",
    )
