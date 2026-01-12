# ==========================================
#         BATTLE CITY AI CONFIG
# ==========================================

# --- SYSTEM & HARDWARE ---
# Number of parallel environments. 
# On Colab (2 cores), keep this around 8-16. 
# On powerful PC (24 cores), go for 48-64.
import multiprocessing
import os
# NUM_CPU = multiprocessing.cpu_count()
NUM_CPU = 8 # Reduced for Hybrid Mode (Transformer + LSTM)
HEADLESS_MODE = True 

# --- TRAINING DURATION ---
TOTAL_TIMESTEPS = 100_000_000 # Forever.

# --- STACK_SIZE = 4 # <--- Minimal Stack. LSTM handles the "Time" memory now.
FRAME_SKIP    = 4     
USE_VISION    = False 
ROM_PATH      = "BattleCity.nes"

USE_RECURRENT   = False    # <--- Disable LSTM
USE_TRANSFORMER = True     # <--- Enable Transformer (Memory Attention)

# --- SAVING ---
CHECKPOINT_FREQ = 50_000 
MODEL_DIR = "models"
LOG_DIR = "logs"

# --- PPO HYPERPARAMETERS ---
LEARNING_RATE = 0.0003   
N_STEPS       = 2048     # Back to standard
BATCH_SIZE    = 4096     # Back to efficient batch size (Stack is small now, so RAM is fine)
N_EPOCHS      = 10      
ENT_COEF      = 0.01     # Less entropy needed with high parallel noise
GAMMA         = 0.99     

# --- SAFETY ---
ALLOW_NEW_MODEL = True # If True, will restart from scratch if no model found. If False, CRASHES.

# --- RESTORED COMPATIBILITY SETTINGS ---
# Required by current train.py/env.py
STACK_SIZE = 4
USE_VISION = False # <--- OPTIMIZATION: RAM ONLY (The Matrix Mode). 5x Speedup.
ROM_PATH = 'BattleCity_fixed.nes'
CLIP_RANGE = 0.2
