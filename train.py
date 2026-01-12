import gymnasium as gym
import os
import time
import math
import cv2
import numpy as np
import pickle
from collections import deque
from stable_baselines3 import PPO
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("Warning: sb3-contrib not installed. RecurrentPPO unavailable.")
    RecurrentPPO = None


from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env # Needed for dynamic kwargs
from stable_baselines3.common.vec_env import VecNormalize # <--- Added
from battle_city_env import BattleCityEnv
import torch as th # Added for Architecture Config

import torch as th # Added for Architecture Config
import config # <--- IMPORT CONFIG

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    :param initial_value: The initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        """
        return progress_remaining * initial_value
    return func

# Network Architecture calculation (Dependent on Config)
# 2048 bytes * STACK_SIZE
first_layer_size = 1024 * config.STACK_SIZE 
if first_layer_size < 512: first_layer_size = 512

class ConsoleLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ConsoleLoggerCallback, self).__init__(verbose)
        self.last_time_steps = 0

    def _on_step(self) -> bool:
        # Print every 1000 steps
        if self.num_timesteps % 1000 == 0:
            mean_rew = "N/A"
            if len(self.model.ep_info_buffer) > 0:
                mean_rew = f"{sum([ep['r'] for ep in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer):.2f}"
            print(f"[{self.num_timesteps} steps] Mean Reward: {mean_rew} (Playing...)", end='\r')
        return True

import traceback

class RenderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.windows_initialized = False # Flag for window setup
        self.history_path = f"{config.MODEL_DIR}/score_history.pkl"
        
        # HEADLESS AUTO-DETECT
        # 1. Config override
        # 2. 'DISPLAY' env var missing (Linux/Colab)
        if getattr(config, 'HEADLESS_MODE', False) or (os.name == 'posix' and 'DISPLAY' not in os.environ):
             print("[System] HEADLESS MODE DETECTED (Colab/Server). GUI Disabled.")
             self.windows_initialized = "HEADLESS" # Permanently disable rendering
        
        # Load History if exists
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'rb') as f:
                    self.score_history = pickle.load(f)
                print(f"[Graph] Loaded history: {len(self.score_history)} games.")
            except Exception as e:
                print(f"[Graph] Load failed: {e}")
                self.score_history = deque() # Unlimited
        else:
            self.score_history = deque() # Unlimited
        
    def _on_step(self) -> bool:
        try:
            # ... (Existing logic for collecting scores) ...
            # Detect End of Episode to Record Score
            dones = self.locals.get('dones', [False])
            infos = self.locals.get('infos', [{}])
            
            # Check ALL environments for finished games
            any_finished = False
            for i, done in enumerate(dones):
                if done:
                    final_score = infos[i].get('score', 0)
                    # NEW: Store Tuple (Score, Timestep)
                    self.score_history.append((final_score, self.num_timesteps))
                    any_finished = True
            
            if any_finished:
                # Save History (If any game finished)
                try:
                    with open(self.history_path, 'wb') as f:
                        pickle.dump(self.score_history, f)
                except Exception as e:
                    print(f"[Graph] Save failed: {e}")
                
                # --- LOG STATS TO CONSOLE (USER REQUEST) ---
                try:
                    # Extract just scores
                    all_scores = []
                    # Optimization: If history is huge, maybe slice? But 100k is fine.
                    for item in self.score_history:
                        if isinstance(item, (tuple, list)):
                             all_scores.append(float(item[0]))
                        else:
                             all_scores.append(float(item))
                    
                    if all_scores:
                        self.logger.record("rollout/max_score", max(all_scores))
                        self.logger.record("rollout/avg_score_all", np.mean(all_scores))
                        self.logger.record("rollout/last_score", all_scores[-1])
                except Exception:
                    pass

            current_obs = self.locals.get('new_obs')
            
            # --- SCORE GRAPH (SEPARATE WINDOW) ---
            # Update Graph window every step (it's fast)
            if self.windows_initialized == "HEADLESS":
                return True # Skip rendering
                
            try:
                # Create black canvas
                g_w, g_h = 800, 400 
                graph_frame = np.zeros((g_h, g_w, 3), dtype=np.uint8)

                if len(self.score_history) > 1:
                    scores = []
                    steps = []
                    for i, item in enumerate(self.score_history):
                        try:
                            if isinstance(item, tuple) or isinstance(item, list):
                                if len(item) >= 2:
                                    scores.append(float(item[0]))
                                    steps.append(int(item[1]))
                                elif len(item) == 1:
                                    scores.append(float(item[0]))
                                    steps.append(0)
                            else:
                                scores.append(float(item))
                                steps.append(0)
                        except: continue
                        
                    if not scores: return True # Skip if empty after filtering
                    
                    min_s, max_s = min(scores), max(scores)
                    if max_s == min_s: max_s += 1 
                    
                    # Dynamic X-Scale
                    total_points = len(scores)
                    
                    # Colors for Cycles (Rainbow-ish)
                    # BGR format
                    colors = [
                        (0, 0, 255),    # Red
                        (0, 165, 255),  # Orange
                        (0, 255, 255),  # Yellow
                        (0, 255, 0),    # Green
                        (255, 255, 0),  # Cyan
                        (255, 0, 0),    # Blue
                        (255, 0, 255),  # Magenta
                    ]
                    
                    for i in range(1, total_points):
                        p1_val = scores[i-1]
                        p2_val = scores[i]
                        
                        # Determine Color based on Step Count of the SECOND point
                        # Cycle changes every config.N_STEPS
                        step_val = steps[i]
                        cycle_idx = (step_val // config.N_STEPS) % len(colors)
                        line_color = colors[cycle_idx]
                        
                        # Scales
                        x1 = int((i-1) * (g_w / (total_points - 1)))
                        x2 = int(i * (g_w / (total_points - 1)))
                        
                        # Invert Y (0 is top)
                        y1 = int((g_h - 20) - ((p1_val - min_s) / (max_s - min_s)) * (g_h - 40))
                        y2 = int((g_h - 20) - ((p2_val - min_s) / (max_s - min_s)) * (g_h - 40))
                        
                        cv2.line(graph_frame, (x1, y1), (x2, y2), line_color, 2) # Thicker line (2)
                        
                        # Only draw dots if not too crowded
                        if total_points < 100:
                            cv2.circle(graph_frame, (x2, y2), 3, (255, 255, 255), -1)

                    # Stats Text
                    cv2.putText(graph_frame, f"Max Score: {max_s:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    cv2.putText(graph_frame, f"Avg (All {total_points}): {np.mean(scores):.2f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    cv2.putText(graph_frame, f"Last: {scores[-1]:.2f}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    
                    # Current Cycle Info
                    current_cycle = self.num_timesteps // config.N_STEPS
                    cv2.putText(graph_frame, f"Update Cycle: {current_cycle}", (550, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[current_cycle % len(colors)], 2)
                                
                else:
                     cv2.putText(graph_frame, "Waiting for games...", (100, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                cv2.imshow("Score History Graph", graph_frame)
            except Exception as e: 
                # print(f"Graph Error: {e}")
                pass
            
            # One-time Window Setup
            if not self.windows_initialized:
                try:
                     cv2.namedWindow("Battle City AI Training", cv2.WINDOW_AUTOSIZE)
                     cv2.namedWindow("Score History Graph", cv2.WINDOW_AUTOSIZE)
                     cv2.moveWindow("Battle City AI Training", 50, 50)
                     cv2.moveWindow("Score History Graph", 800, 50)
                     self.windows_initialized = True
                except Exception as e:
                    print(f"\n[WARNING] Could not open display: {e}. Switching to HEADLESS MODE (Console only).")
                    self.windows_initialized = "HEADLESS"

            if self.windows_initialized == "HEADLESS":
                return True # Skip rendering
                
            # render() logic handled via INFO to save bandwidth
            # We only render the first environment's view
            try:
                # Get the frame from INFO (bypassing Agent's blindfold)
                frame = infos[0].get('render')
                
                if frame is not None:
                    # Resize for Window (84x84 -> 672x672)
                    frame_img = frame.astype('uint8')
                    frame_big = cv2.resize(frame_img, (672, 672), interpolation=cv2.INTER_NEAREST)
                    
                    # Convert to BGR for Colored Text
                    display_frame = cv2.cvtColor(frame_big, cv2.COLOR_GRAY2BGR)
                    
                    # Draw HUD
                    kills = infos[0].get('kills', 0)
                    score = infos[0].get('score', 0.0)
                    total_steps = self.num_timesteps
                    
                    # Top HUD
                    cv2.putText(display_frame, f"KILLS: {kills}", (20, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2) # Red
                    cv2.putText(display_frame, f"SCORE: {score:.1f}", (400, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2) # Green
                    cv2.putText(display_frame, f"Steps: {total_steps}", (20, 650), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) 
                    
                    cv2.imshow("Battle City AI Training", display_frame)
                    cv2.waitKey(1) # 1ms delay
                    
            except Exception as render_err:
                 # print(f"Render Error: {render_err}")
                 pass
        except Exception as e:
            if self.num_timesteps % 1000 == 0:
                print(f"\n[DEBUG] CV2 Render Error: {e}")
            pass
        return True

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, SeqLen, Dim]
        # Add position encoding to the embedding
        return x + self.pe[:, :x.size(1), :]

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        # Observation space is (N_STACK * RAM_SIZE) flattened
        # We assume input is (Batch, STACK_SIZE * RAM_SIZE)
        
        super().__init__(observation_space, features_dim)
        
        self.stack_size = config.STACK_SIZE
        # self.ram_size = 2048 # Fixed NES RAM (DEPRECATED)
        # Calculate Input Feature Size dynamically (e.g., 32 or 2048)
        self.ram_size = observation_space.shape[0] // self.stack_size
        self.d_model = 512   # Embedding Size
        self.nhead = 8
        self.num_layers = 4
        
        # Project RAM (2048) -> Embedding (512)
        self.input_net = nn.Sequential(
            nn.Linear(self.ram_size, self.d_model),
            nn.ReLU()
        )
        
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.stack_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input: (Batch, STACK_SIZE * RAM_SIZE)
        batch_size = observations.shape[0]
        
        # Reshape to (Batch, Stack, RAM)
        # Verify shape logic: SB3 flattens inputs.
        x = observations.view(batch_size, self.stack_size, self.ram_size)
        
        # Project to Embedding
        x = self.input_net(x) # (Batch, Stack, 512)
        
        # Add Position
        x = self.pos_encoder(x)
        
        # Transformer Pass
        x = self.transformer_encoder(x) # (Batch, Stack, 512)
        
        # Aggregation: Take the LAST vector (The most recent frame, enriched by history)
        # The sequence is usually [Oldest ... Newest] 
        # In our env implementation: append() adds to end. So -1 is newest.
        features = x[:, -1, :] # (Batch, 512)
        
        return features

def train():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    print(f"--- BATTLE CITY AI TRAINING (VISUAL MODE) ---")
    
    # Use SubprocVecEnv for Multicore Speed
    # Windows Note: SubprocVecEnv requires non-lambda functions usually. 
    # We use a list comprehension of factory functions.
    
    # Pass Config to Environment
    env_kwargs = {'use_vision': config.USE_VISION, 'stack_size': config.STACK_SIZE}
    
    if config.NUM_CPU > 1:
        # We need to import stable_baselines3.common.env_util to use make_vec_env with SubprocVecEnv correctly if not done automatically
        env = make_vec_env(BattleCityEnv, n_envs=config.NUM_CPU, seed=42, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    else:
        env = make_vec_env(BattleCityEnv, n_envs=1, seed=42, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs)

    # Apply VecNormalize (Stabilizes Training)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Load or Create Model
    # Choose Policy Type based on Config
    policy_type = "MultiInputPolicy" if config.USE_VISION else "MlpPolicy"
    
    print(f"Stats: Vision={config.USE_VISION}, Stack={config.STACK_SIZE}, Policy={policy_type}")
    # Auto-Resume Logic
    latest_model_path = f"{config.MODEL_DIR}/battle_city_interrupted.zip"
    final_model_path = f"{config.MODEL_DIR}/battle_city_final.zip"
    
    # Determine Learning Rate
    if hasattr(config, 'LR_START'):
        lr_schedule = linear_schedule(config.LR_START)
        print(f"Using Linear LR Schedule starting at {config.LR_START}")
    else:
        lr_schedule = config.LEARNING_RATE
        print(f"Using Constant Learning Rate: {config.LEARNING_RATE}")

    # Select Model Class
    if getattr(config, 'USE_RECURRENT', False) and RecurrentPPO is not None:
        ModelClass = RecurrentPPO
        print("Model Class: RecurrentPPO (LSTM)")
    else:
        ModelClass = PPO
        print("Model Class: Standard PPO")

    if os.path.exists(latest_model_path):
        print(f"Loading interrupted model from {latest_model_path}...")
        try:
            model = ModelClass.load(latest_model_path, env=env)
            model.learning_rate = lr_schedule # Update schedule
            model.ent_coef = getattr(config, 'ENTROPY_COEF', getattr(config, 'ENT_COEF', 0.01))
            model.clip_range = lambda _: config.CLIP_RANGE
            reset_timesteps = False
        except Exception as e:
            print(f"Failed to load model architecture mismatch? Error: {e}")
            print("Starting FRESH model due to incompatibility.")
            model = None # Trigger creation logic
            reset_timesteps = True
            
    elif os.path.exists(final_model_path):
        print(f"Loading existing model from {final_model_path}...")
        try:
            model = ModelClass.load(final_model_path, env=env)
            model.learning_rate = lr_schedule # Update schedule
            model.ent_coef = getattr(config, 'ENTROPY_COEF', getattr(config, 'ENT_COEF', 0.01))
            model.clip_range = lambda _: config.CLIP_RANGE
            reset_timesteps = False
        except Exception as e:
             print(f"Failed to load model (mismatch?). Error: {e}")
             model = None
             reset_timesteps = True
    else:
        model = None
        reset_timesteps = True

    # --- TRANSFORMER ARCHITECTURE ---
    # Moved to Global Scope


    # --- MODEL CREATION ---
    if model is None:
        if getattr(config, 'USE_TRANSFORMER', False):
            # HYBRID MODE: Tansformer + LSTM
            if getattr(config, 'USE_RECURRENT', False): 
                 print("Creating NEW MODEL (Hybrid: Transformer + RecurrentPPO)...")
                 if RecurrentPPO is None:
                     print("CRITICAL ERROR: sb3-contrib not installed. Cannot use LSTM.")
                     return

                 policy_kwargs = dict(
                    features_extractor_class=TransformerFeatureExtractor,
                    features_extractor_kwargs=dict(features_dim=512),
                    net_arch=dict(pi=[512, 256], vf=[512, 256]), 
                    activation_fn=th.nn.ReLU,
                    lstm_hidden_size=512, # LSTM Layer size
                    n_lstm_layers=1       # 1 Layer is enough for hybrid
                 )
                 
                 # LSTM Policies
                 if config.USE_VISION:
                     lstm_policy_type = "MultiInputLstmPolicy"
                 else:
                     lstm_policy_type = "MlpLstmPolicy"
                     
                 model = RecurrentPPO(
                    lstm_policy_type, 
                    env, 
                    verbose=1,
                    tensorboard_log=config.LOG_DIR,
                    learning_rate=lr_schedule,
                    n_steps=config.N_STEPS,
                    batch_size=config.BATCH_SIZE,
                    n_epochs=getattr(config, 'N_EPOCHS', 10),
                    ent_coef=getattr(config, 'ENTROPY_COEF', getattr(config, 'ENT_COEF', 0.01)),
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=config.CLIP_RANGE,
                    device="cuda",
                    policy_kwargs=policy_kwargs 
                 )

            # STANDARD TRANSFORMER (No LSTM)
            else:
                 print(f"Creating NEW MODEL (Transformer PPO)... STACK_SIZE={config.STACK_SIZE}")
                 policy_kwargs = dict(
                    features_extractor_class=TransformerFeatureExtractor,
                    features_extractor_kwargs=dict(features_dim=512),
                    net_arch=dict(pi=[512, 256], vf=[512, 256]), 
                    activation_fn=th.nn.ReLU,
                 )
    
                 model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log=config.LOG_DIR,
                    learning_rate=lr_schedule, # Use the determined LR schedule
                    n_steps=config.N_STEPS,
                    batch_size=config.BATCH_SIZE,
                    n_epochs=getattr(config, 'N_EPOCHS', 10),
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=config.CLIP_RANGE,
                    ent_coef=getattr(config, 'ENTROPY_COEF', getattr(config, 'ENT_COEF', 0.01)),
                    policy_kwargs=policy_kwargs,
                    device="cuda"
                 )
        else: # Original PPO/RecurrentPPO creation logic
            # Define Network Architecture (Optimized for High-RAM Colab)
            # OLD: [512, 256] -> Good.
            # NEW: [1024, 512] -> Smarter. We have 15GB VRAM, let's use it.
            
            policy_kwargs = dict(
                activation_fn=th.nn.ReLU,
                net_arch=dict(pi=[1024, 512], vf=[1024, 512]),
                lstm_hidden_size=1024, # <--- KEEP HUGE MEMORY
                n_lstm_layers=2,       # <--- STABLE DEPTH (Gold Standard)
                shared_lstm=False, 
                enable_critic_lstm=True
            )

            # Check for RecurrentPPO (LSTM)
            if getattr(config, 'USE_RECURRENT', False) and RecurrentPPO is not None:
                 print("Creating NEW MODEL (RecurrentPPO / LSTM)...")
                 # LSTM Policies
                 if config.USE_VISION:
                     lstm_policy_type = "MultiInputLstmPolicy"
                 else:
                     lstm_policy_type = "MlpLstmPolicy"
                 
                 model = RecurrentPPO(
                    lstm_policy_type, 
                    env, 
                    verbose=1,
                    tensorboard_log=config.LOG_DIR,
                    learning_rate=lr_schedule,
                    n_steps=config.N_STEPS,
                    batch_size=config.BATCH_SIZE,
                    n_epochs=getattr(config, 'N_EPOCHS', 10),
                    ent_coef=getattr(config, 'ENTROPY_COEF', getattr(config, 'ENT_COEF', 0.01)),
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=config.CLIP_RANGE,
                    device="cuda",
                    policy_kwargs=policy_kwargs 
                 )
            else:
                 # Standard PPO
                 print(f"Creating NEW MODEL ({policy_kwargs['net_arch']['pi'][0]}x{policy_kwargs['net_arch']['pi'][1]})...")
                 model = PPO(
                    policy_type, 
                    env, 
                    verbose=1,
                    tensorboard_log=config.LOG_DIR,
                    learning_rate=lr_schedule,
                    n_steps=config.N_STEPS,          
                    batch_size=config.BATCH_SIZE,       
                    n_epochs=getattr(config, 'N_EPOCHS', 10),
                    ent_coef=getattr(config, 'ENTROPY_COEF', getattr(config, 'ENT_COEF', 0.01)),
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=config.CLIP_RANGE,
                    device="cuda",
                    policy_kwargs=policy_kwargs
                 )
        reset_timesteps = True

    checkpoint_callback = CheckpointCallback(
        save_freq=config.CHECKPOINT_FREQ,
        save_path=config.MODEL_DIR,
        name_prefix="battle_city_ppo"
    )
    
    logger_callback = ConsoleLoggerCallback()
    render_callback = RenderCallback()

    print("Start VISUAL Learning... (Press Ctrl+C to stop)")
    try:
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, logger_callback, render_callback],
            progress_bar=True,
            reset_num_timesteps=reset_timesteps
        )
        model.save(f"{config.MODEL_DIR}/battle_city_final")
        print("Training Finished!")
        
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            print("\n[User] Stopped by Keyboard (Ctrl+C).")
        else:
            print(f"\n[CRITICAL] Training Interrupted/Crashed: {e}")
            
        print("Saving EMERGENCY model...")
        print("Saving EMERGENCY model...")
        model.save(f"{config.MODEL_DIR}/battle_city_interrupted")
        print("Saved.")
        print("Saved.")
        
    finally:
        print("Closing environment...")
        try:
             env.close()
        except (EOFError, BrokenPipeError, ConnectionResetError):
             # These are normal during forced shutdown of subprocesses
             pass
        except Exception as e:
             print(f"Cleanup Warnings: {e}")

if __name__ == "__main__":
    train()
