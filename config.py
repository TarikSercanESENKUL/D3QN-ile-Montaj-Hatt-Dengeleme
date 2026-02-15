import os

# --- Simulation Constants ---
SIM_TIME = 30 * 24 * 60  # Simulation time in minutes (e.g., 30 days)
SHIFT_DURATION = 480     # 8.0 hours * 60 minutes = 480 (Includes 30 min break)
NUM_SHIFTS = 3           # 3 active shifts per day
WORKERS_PER_SHIFT = 46   # Number of workers per shift (Approx to match constraints)
TOTAL_WORKERS = 184      # (3 shifts + 1 reserve) * 46
ABSENTEEISM_RATE = 0.15  # 15% probability based on heavy industry data
PART_SHORTAGE_RATE = 0.05 # 5% probability of missing part (Line Feeding Delay)

# Standard Caps
STATION_CAPACITIES = {
    "R1": 6, "R10": 6, "R11": 6, "R12": 6, 
    "R2": 4, "R3": 4, "R4": 1, "R5": 4, 
    "R6": 1, "R7": 1, "R8": 1, "R9": 6,
    "G1": 6, "G2": 6 
}

STATIONS = list(STATION_CAPACITIES.keys())
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_SIZE = 50000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 2000 # Higher value = Slower, Smoother decay for Exponential
TARGET_UPDATE = 2000     # Update target network LESS frequently for stability

# --- Reward System ---
REWARD_COMPLETION = 10.0   # Reward for completing a bus (G2 - Final Station)
REWARD_STEP = -0.1         # Step penalty (to minimize makespan)
REWARD_OVERTIME = -50.0    # Penalty for exceeding shift duration (not strictly applicable in continuous, but for KPIs)
REWARD_IDLE = -0.5         # Penalty if a worker is idle while tasks are waiting

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ROTA_FILE = os.path.join(DATA_DIR, 'factory_route.csv')
BOM_FILE = os.path.join(DATA_DIR, 'parts_bom.csv')
SKILLS_FILE = os.path.join(DATA_DIR, 'worker_skills.csv')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# --- Station Mapping ---
# STATIONS is now defined above with CAPACITIES
