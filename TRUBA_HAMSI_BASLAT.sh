#!/bin/bash
#SBATCH -p hamsi
#SBATCH -J DQN_Bus_Factory
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "------------------------------------------------"
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "------------------------------------------------"

# --- 1. MODULE LOADING ---
# Try to load a standard python module available on Truba
# If these specific versions aren't found, the script might fail, 
# so we try multiple common ones or rely on system python if valid.
echo "Loading Python Module..."
module load apps/python/3.9.1 2>/dev/null || module load facebook/anaconda3/2020.07 2>/dev/null || echo "Module load skipped or failed, using system python"

# --- 2. ENVIRONMENT SETUP (Auto-Create) ---
VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' not found. Creating it now..."
    python3 -m venv $VENV_DIR
    
    echo "Activating '$VENV_DIR'..."
    source $VENV_DIR/bin/activate
    
    echo "Upgrading pip and installing dependencies..."
    pip install --upgrade pip
    
    # Check if requirements.txt exists, otherwise install manually
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "requirements.txt not found. Installing default packages..."
        pip install numpy pandas simpy gymnasium torch
    fi
    
    echo "Environment setup complete."
else
    echo "Virtual environment '$VENV_DIR' found. Activating..."
    source $VENV_DIR/bin/activate
fi

# --- 3. EXECUTION ---
echo "Starting Training..."
echo "Python path: $(which python)"

# Run training with unbuffered output
python -u train.py --episodes 500


echo "Generating visualizations..."
python generate_visualization.py

echo "Performing Statistical Analysis..."
python perform_statistical_analysis.py

echo "Running Stress Test (Hard Mode - 1 Month)..."
python run_stress_test.py

echo "------------------------------------------------"
echo "Job finished on $(date)"
