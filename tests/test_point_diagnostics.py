import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.diagnostics import diagnose_base_points  # noqa: E402
from utils.transformations import poincare_to_pseudo_spherical  # noqa: E402


def test_point_diagnostics_reports_sources_and_validity():
    poincare_points = np.array(
        [
            [0.05, 0.0, 0.0],  # firmly inside
            [0.2, 0.0, 0.0],   # near artificial boundary
        ],
        dtype=float,
    )
    pseudos = poincare_to_pseudo_spherical(poincare_points)
    metadata = {"dirichlet_count": 1, "fallback_count": 1}

    # Provide synthetic Dirichlet references so we can force a violation
    dirichlet_refs = [np.array([0.205, 0.0, 0.0])]

    payload = diagnose_base_points(
        poincare_points,
        pseudos,
        manifold_name="test_manifold",
        group_elements=[],
        sampling_metadata=metadata,
        dirichlet_images=dirichlet_refs,
    )

    assert payload["summary"]["dirichlet_points"] == 1
    assert payload["summary"]["fallback_points"] == 1
    assert payload["points"][0]["source"] == "dirichlet"
    assert payload["points"][1]["source"] == "fallback"
    assert payload["points"][0]["valid"] is True
    assert payload["points"][1]["valid"] is False
    assert payload["summary"]["invalid_points"] == 1
