#!/bin/bash
#SBATCH -p barbun           # Partition name (barbun, akya-cuda, etc.)
#SBATCH -A your_account     # Account name (optional, replace if needed)
#SBATCH -J DQN_Bus_Factory  # Job name
#SBATCH -o logs/%j_output.txt # Output log file
#SBATCH -e logs/%j_error.txt  # Error log file
#SBATCH -N 1                # Number of nodes
#SBATCH -n 1                # Number of tasks
#SBATCH -c 10               # Number of CPUS per task
#SBATCH --time=24:00:00     # Time limit

echo "Job started at: $(date)"
echo "Node: $(hostname)"

# Load necessary modules (Adjust based on Truba environment)
module load apps/anaconda3/2021.05 || module load facebook/anaconda3/2020.07

# Activate the environment
source activate dqn_bus_env || conda activate dqn_bus_env

# Ensure log directory exists
mkdir -p logs
mkdir -p checkpoints

# Run the training script
python train.py --episodes 5000

echo "Job finished at: $(date)"
