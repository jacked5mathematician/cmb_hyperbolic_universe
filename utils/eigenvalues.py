from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .conventions import q_squared

MULTIPLICITY_TOL = 1.1
POLY_EPS = 1e-12


def _local_minima(values: Sequence[float]) -> List[int]:
    mins: List[int] = []
    for i in range(1, len(values) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            mins.append(i)
    return mins


def _refine_minimum(k_values: np.ndarray, chi_values: np.ndarray, idx: int) -> Tuple[float, float]:
    if idx <= 0 or idx >= len(k_values) - 1:
        return float(k_values[idx]), float(chi_values[idx])
    ks = k_values[idx - 1 : idx + 2]
    chis = chi_values[idx - 1 : idx + 2]
    coeffs = np.polyfit(ks, chis, 2)
    if abs(coeffs[0]) < POLY_EPS:
        return float(k_values[idx]), float(chi_values[idx])
    k_star = -coeffs[1] / (2 * coeffs[0])
    chi_star = np.polyval(coeffs, k_star)
    return float(k_star), float(chi_star)


def extract_eigenvalues_from_spectrum(
    spectrum_path: Path | str,
    output_dir: Path | str,
    threshold: float | None = None,
    refine: bool = True,
) -> Path:
    """
    Load spectrum.npz, detect local minima, and write eigenvalues.csv with
    columns (k, q2, chi2, multiplicity_hint).
    """
    spectrum_path = Path(spectrum_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(spectrum_path)
    k_values = data["k_values"]
    chi1 = data["chi2_rank_1"]
    rank_curves = [chi1]
    for r in range(2, 6):
        key = f"chi2_rank_{r}"
        if key in data:
            rank_curves.append(data[key])

    minima = _local_minima(chi1)
    rows: List[Tuple[float, float, float, int]] = []
    for idx in minima:
        chi_val = float(chi1[idx])
        if not np.isfinite(chi_val):
            continue
        if threshold is not None and chi_val > threshold:
            continue

        k_refined, chi_refined = _refine_minimum(k_values, chi1, idx) if refine else (float(k_values[idx]), chi_val)
        multiplicity_hint = 0
        for curve in rank_curves:
            if idx < len(curve) and np.isfinite(curve[idx]) and curve[idx] <= chi_val * MULTIPLICITY_TOL:
                multiplicity_hint += 1
        rows.append((k_refined, q_squared(k_refined), float(chi_refined), multiplicity_hint))

    csv_path = output_dir / "eigenvalues.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "q2", "chi2", "multiplicity_hint"])
        for row in rows:
            writer.writerow(row)

    return csv_path


__all__ = ["extract_eigenvalues_from_spectrum"]
