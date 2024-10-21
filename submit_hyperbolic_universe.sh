#!/bin/bash
#SBATCH --job-name=spectrumjob            # Job name
#SBATCH --output=output_%A_%a.log         # Output log, %A is job ID, %a is array task ID
#SBATCH --time=02:00:00                   # Time limit hrs:min:sec
#SBATCH --nodes=1                         # Number of nodes per task
#SBATCH --ntasks=1                        # Number of tasks per job
#SBATCH --cpus-per-task=128                # Adjust based on your code's threading (e.g., 32 cores per node)
#SBATCH --mem=128G                        # Memory per node
#SBATCH --partition=regularshort          # Partition or queue
#SBATCH --array=0-7                       # Job array index (0 to num_chunks - 1)
#SBATCH --mail-type=BEGIN,END,FAIL        # Notifications
#SBATCH --mail-user=ercetin.utku@gmail.com  # Your email address

# Load necessary modules
module load python/3.10.8
module load texlive

# Activate your Python virtual environment
source /home2/s4629701/myenv/bin/activate

# Run your Python script with the chunk index
srun python /home2/s4629701/cmb_hyperbolic_universe/main.py --chunk_index=$SLURM_ARRAY_TASK_ID --num_chunks=8