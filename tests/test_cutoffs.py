import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MAX_FALLBACK_FRACTION = 0.5


def test_m188_cutoff_fallback_fraction(tmp_path: Path):
    """Ensure cutoff fallback is not 1.0 for a small k sweep on m188(-1,1)."""
    pytest.importorskip("snappy", reason="SnapPy required for cutoff regression")
    from main import run_pipeline

    # Small sweep keeps runtime reasonable while catching regressions for k=1..3
    k_values = np.arange(1.0, 4.0, 1.0)
    output_dir = tmp_path / "m188_cutoffs"
    result = run_pipeline(
        manifold_name="m188(-1,1)",
        k_values=k_values,
        n_points=6,
        seed=123,
        small_test=True,
        dry_run=False,
        output_dir=output_dir,
        chi2_mode="paper",
    )

    data = np.load(result["spectrum_path"])
    fallback_used = data["fallback_used"]

    assert len(fallback_used) == len(k_values)
    # Regression: cutoff fallback fraction should drop below unity
    assert fallback_used.mean() < MAX_FALLBACK_FRACTION
