#!/bin/bash
#SBATCH --job-name=spectrumjob      # Job name, describe what your job does
#SBATCH --output=output_%j.log                  # Standard output and error log, %j will add the job ID
#SBATCH --time=06:00:00                         # Time limit hrs:min:sec (adjust this based on your expected runtime)
#SBATCH --nodes=1                               # Number of nodes (1 node should be sufficient unless you need more)
#SBATCH --ntasks=1                              # Number of tasks (1 for serial or single process)
#SBATCH --cpus-per-task=128                   # Number of CPU cores per task (adjust based on how parallelized your code is)
#SBATCH --mem=128G                               # Memory per node (adjust based on memory needs, 64 GB here)
#SBATCH --partition=regularlong                    # Partition or queue (use an appropriate partition available in your cluster)
#SBATCH --mail-type=BEGIN,END,FAIL                    # Notifications for job done or failed
#SBATCH --mail-user=u.ercetin@student.rug.nl       # Your email address for job notifications

# Load necessary modules
module load python/3.10.8                        # Adjust the Python version to match what you're using

# Activate your Python virtual environment
source /home2/s4629701/myenv/bin/activate

# Run your Python script
srun python /home2/s4629701/cmb_hyperbolic_universe/main.py
