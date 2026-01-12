import gymnasium as gym
import numpy as np
from battle_city_env import BattleCityEnv
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

# --- 1. DATA COLLECTION ---
def collect_data(num_samples=2000):
    print(f"Collecting {num_samples} samples of (RAM -> Action)...")
    env = BattleCityEnv(render_mode='human', use_vision=False)
    env.reset()
    
    data_x = [] # RAM
    data_y = [] # Action
    
    obs, _ = env.reset()
    
    # Simple Heuristic/Random Policy to get varied data
    # We want valid gameplay, so random is okay-ish, but bias towards movement helps.
    
    for _ in range(num_samples):
        # 20% Chance of FIRE (Action 5)
        # 80% Move
        if np.random.rand() < 0.2:
            action = 5 # Fire
        else:
            action = np.random.randint(1, 5) # 1=Up, 2=Down, 3=Left, 4=Right
            
        # Execute
        env.step(action)
        # env.render() # Optional: Speed up by disabling
        
        # Record state AFTER action (What does the RAM look like doing this?)
        # Actually... we want: RAM State -> Policy Decision.
        # But here we are reverse engineering "What defines this state?"
        # Let's train: Input(RAM) -> Predicts(Action that WAS taken or IS optimal?)
        
        # Better approach for semantics:
        # "Which RAM bits CHANGE when I do X?" (We did this in log)
        # This Neural Network approach: 
        # "Which RAM bits correlate with specific actions being valid/active?"
        
        # Let's stick to "Input: RAM -> Output: Predicted Action"
        # If the model learns that "When RAM[0x51]==2, we often Go Right", it's a weak signal.
        # But if we train it on *changes*:
        # "Input: RAM_Now - RAM_Prev -> Output: Action"
        # This is strictly causal. Action caused Change.
        
        ram = np.array(env.raw_env.ram[:2048], dtype=np.float32) / 255.0
        data_x.append(ram)
        data_y.append(action)
        
        if _ % 100 == 0:
            env.reset()
            
    env.close()
    return np.array(data_x), np.array(data_y)

# --- 2. ANALYZER MODEL ---
class RamAnalyzer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RamAnalyzer, self).__init__()
        # Linear layer = Logic Gates. Weights directly show importance.
        self.classifier = nn.Linear(input_dim, output_dim) 
        
    def forward(self, x):
        return self.classifier(x)

def train_and_analyze():
    # 1. Get Data
    X_raw, y_raw = collect_data(3000)
    
    # 2. Prepare Tensors
    X = torch.FloatTensor(X_raw)
    y = torch.LongTensor(y_raw)
    
    input_dim = 2048
    output_dim = 6 # NOOP, UP, DOWN, LEFT, RIGHT, FIRE
    
    model = RamAnalyzer(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Training Analyzer Network...")
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    # 3. EXTRACT SEMANTICS (Feature Importance)
    print("\nanalysis: 'What RAM addresses matter for each Action?'")
    print("========================================================")
    
    # Weights shape: [Classes, Inputs] -> [6, 2048]
    weights = model.classifier.weight.detach().numpy()
    
    actions = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT", "FIRE"]
    
    for action_idx, action_name in enumerate(actions):
        if action_idx == 0: continue # Skip NOOP
        
        # Get absolute weights for this class
        # (Positive weight = Presence of value High encourages this action)
        # (Negative weight = Presence of value High discourages this action)
        # We look for MAX ABSOLUTE weight.
        action_weights = weights[action_idx]
        
        # Get Top 5 Indices
        top_indices = np.argsort(np.abs(action_weights))[-5:][::-1]
        
        print(f"\nACTION: {action_name}")
        for idx in top_indices:
            weight_val = action_weights[idx]
            print(f"  - Addr 0x{idx:04X} ({idx}): Weight {weight_val:.4f}")
            
            # Simple interpretation
            if idx == 0x51: print("    -> (Known: Player Lives)")
            if idx == 0x92: print("    -> (Known: Game State)")
            if idx in [0x2E, 0x2F]: print("    -> (Known: Player Pos)")
            
    print("\nDone. If unknown addresses appear with high weights, check them in the Hex Editor!")

if __name__ == "__main__":
    train_and_analyze()
