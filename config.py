# BATTLE CITY AI - CONFIGURATION FILE

# --- ENVIRONMENT SETTINGS ---
USE_VISION = True   # Set True to enable Vision (Pixels) + RAM. False = RAM Only (Faster).
STACK_SIZE = 4       # Number of Frames/RAM-dumps to stack (Temporal Context). 
                     # Higher = More "Memory" of movement, but bigger inputs.

# --- TRAINING SETTINGS ---
NUM_CPU = 24         # Number of parallel environments (Processes).
                     # Recommended: Count of Logical Processors on your CPU.
                     
TOTAL_TIMESTEPS = 5_000_000  # Total frames to train for.
CHECKPOINT_FREQ = 10_000     # Save model every N steps.

# --- HYPERPARAMETERS ---
# LEARNING_RATE_SCHEDULE with linear decay
LR_START = 0.0003       # Start: Fast learning
LR_END = 0.0            # End: Micro-tuning (effectively 0)
ENTROPY_COEF = 0.05     # 0.05: Balanced exploration. Tries new things but doesn't ignore rewards.
CLIP_RANGE = 0.1        # 0.1: Less "jumping". More conservative updates.
BATCH_SIZE = 512        # 512: Averaging more examples to smooth out noise.
N_STEPS = 2048          # 2048: Standard deep RL buffer. Much more stable updates.

# --- PATHS ---
MODEL_DIR = "models"
LOG_DIR = "logs"
ROM_PATH = 'BattleCity_fixed.nes'
