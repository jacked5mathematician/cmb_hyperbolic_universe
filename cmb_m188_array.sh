#!/bin/bash
#SBATCH --job-name=cmb_m188
#SBATCH --partition=regular
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-79
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# Habrok note: modules are NOT inherited by jobs; load what you need here.
module load Python/3.11

cd "$SLURM_SUBMIT_DIR"

# Activate your venv (must exist in the repo directory)
source .venv/bin/activate

# Avoid oversubscription on 128-core nodes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Optional fail-fast sanity check (good for first run; remove later for speed)
python -c "import snappy; print('SnapPy OK')"

MANIFOLD="m188(-1,1)"
K_MIN="1.0"
K_MAX="10.0"
NUM_K="400"

N_POINTS="60"
WORD_DEPTH="3"

CHUNK_INDEX="${SLURM_ARRAY_TASK_ID}"
NUM_CHUNKS=$((SLURM_ARRAY_TASK_MAX + 1))

RUN_TAG="m188_k1_10_numk${NUM_K}_wd${WORD_DEPTH}_np${N_POINTS}"
OUTDIR="runs/${RUN_TAG}"

mkdir -p "${OUTDIR}" logs

python -m main \
  --manifold "${MANIFOLD}" \
  --k-min "${K_MIN}" --k-max "${K_MAX}" --num-k "${NUM_K}" \
  --n-points "${N_POINTS}" \
  --word-depth "${WORD_DEPTH}" \
  --k-chunk-index "${CHUNK_INDEX}" \
  --k-num-chunks "${NUM_CHUNKS}" \
  --output-dir "${OUTDIR}" \
  --require-snappy