import gymnasium as gym
import os
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from battle_city_env import BattleCityEnv
import config


import math
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
        self.ram_size = 2048 # Fixed NES RAM
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

def test():
    print("--- BATTLE CITY AI - TESTING MODE ---")
    
    # Setup Environment
    # We use DummyVecEnv for a single environment interaction
    env_kwargs = {'use_vision': config.USE_VISION, 'stack_size': config.STACK_SIZE}
    env = DummyVecEnv([lambda: BattleCityEnv(**env_kwargs)])

    # Load Model
    import sys
    
    # Default path
    model_path = f"{config.MODEL_DIR}/battle_city_final.zip"
    
    # Check CLI args
    if len(sys.argv) > 1:
        custom_path = sys.argv[1]
        if os.path.exists(custom_path):
            model_path = custom_path
        else:
            print(f"Warning: Custom path {custom_path} not found. Using default.")
    
    if not os.path.exists(model_path):
        model_path = f"{config.MODEL_DIR}/battle_city_interrupted.zip"
        if not os.path.exists(model_path):
            print(f"Error: No model found in {config.MODEL_DIR}")
            return

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=env)

    obs = env.reset()
    
    cv2.namedWindow("Battle City AI Test", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Battle City AI Test", 50, 50)

    print("Playing... Press 'q' or Ctrl+C to stop.")

    try:
        while True:
            # Predict action
            # deterministic=True makes the agent pick the best action (no exploration noise)
            action, _states = model.predict(obs, deterministic=True)
            
            obs, rewards, dones, infos = env.step(action)
            
            # Rendering
            try:
                frame = infos[0].get('render')
                if frame is not None:
                    # Resize for visibility (84x84 -> 672x672)
                    frame_img = frame.astype('uint8')
                    frame_big = cv2.resize(frame_img, (672, 672), interpolation=cv2.INTER_NEAREST)
                    
                    # Convert to BGR
                    display_frame = cv2.cvtColor(frame_big, cv2.COLOR_GRAY2BGR)
                    
                    # Add Info Text
                    kills = infos[0].get('kills', 0)
                    score = infos[0].get('score', 0.0)
                    
                    cv2.putText(display_frame, f"KILLS: {kills}", (20, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.putText(display_frame, f"SCORE: {score:.1f}", (400, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                    cv2.imshow("Battle City AI Test", display_frame)
                    
                    # Wait 20ms (approx 50 FPS)
                    if cv2.waitKey(20) & 0xFF == ord('q'):
                        break
            except Exception as e:
                print(f"Render Error: {e}")

            if dones[0]:
                obs = env.reset()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test()
