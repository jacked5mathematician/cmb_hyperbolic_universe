#!/bin/bash
#SBATCH --job-name=cmb_m188_combine
#SBATCH --partition=regular
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load Python/3.11
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

# Ensure repo-root imports work for scripts/ invoked as files
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUN_TAG="m188_k1_10_numk400_wd3_np60"
OUTDIR="runs/${RUN_TAG}"

python scripts/combine_spectra.py \
  --input-dir "${OUTDIR}" \
  --output "${OUTDIR}/spectrum.npz"

python scripts/extract_eigenvalues.py \
  --spectrum "${OUTDIR}/spectrum.npz" \
  --output-dir "${OUTDIR}"

python - <<'PY'
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

outdir = Path("runs/m188_k1_10_numk400_wd3_np60")
d = np.load(outdir/"spectrum.npz")
k = d["k_values"]

plt.figure(figsize=(9,5))
plt.plot(k, d["chi2_rank_1"], label="Rank 1")
for r in range(2, 6):
    key = f"chi2_rank_{r}"
    if key in d:
        plt.plot(k, d[key], alpha=0.5, label=f"Rank {r}")

plt.yscale("log")
plt.xlabel("k")
plt.ylabel("chi^2")
plt.title("m188(-1,1) chi^2 spectrum (merged)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(outdir/"chi2_spectrum_merged.png", dpi=300)
print("Wrote", outdir/"chi2_spectrum_merged.png")
PY