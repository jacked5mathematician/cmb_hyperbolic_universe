#!/bin/bash
#SBATCH --job-name=spectrumjob
#SBATCH --output=output_%j.log
#SBATCH --time=00:30:00
#SBATCH --nodes=4                   # Number of nodes
#SBATCH --ntasks-per-node=1         # One task per node
#SBATCH --cpus-per-task=128         # CPUs per node (adjust as needed)
#SBATCH --mem=128G
#SBATCH --partition=regularshort
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ercetin.utku@gmail.com

# Load necessary modules
module load texlive

# Activate your Python virtual environment
source /home2/s4629701/myenv/bin/activate

# Set environment variables
export NUM_NODES=4                  # Total number of nodes (matches --nodes)
export CHUNKS_PER_NODE=100          # Number of chunks per node (adjust as needed)
export NUM_JOBS=100                # Number of parallel jobs per node (should be <= cpus-per-task)
export OMP_NUM_THREADS=1            # Number of threads per process (usually 1 when using joblib)

# Run your main Python script across all nodes
srun --nodes=$NUM_NODES --ntasks=$NUM_NODES --ntasks-per-node=1 --cpus-per-task=100 \
    python /home2/s4629701/cmb_hyperbolic_universe/main.py --wait

# Run the combiner script after main.py completes on all nodes
srun --nodes=1 --ntasks=1 --cpus-per-task=4 \
    python /home2/s4629701/cmb_hyperbolic_universe/combiner.py