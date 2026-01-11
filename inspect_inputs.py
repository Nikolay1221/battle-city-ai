
import gymnasium as gym
from battle_city_env import BattleCityEnv
import numpy as np
import cv2

def inspect():
    print("--- INSPECTING MODEL INPUTS ---")
    
    # Initialize Env (Single instance, no parallelism needed for check)
    env = BattleCityEnv(render_mode='human', use_vision=True, stack_size=4)
    obs, info = env.reset()
    
    print("\n1. OBSERVATION STRUCTURE")
    print(f"Keys: {obs.keys()}")
    
    # --- SCREEN ---
    screen = obs['screen']
    print(f"\n2. SCREEN INPUT")
    print(f"Shape: {screen.shape} (Channels, Height, Width)")
    print(f"Type: {screen.dtype}")
    print(f"Range: [{np.min(screen)}, {np.max(screen)}]")
    
    # --- RAM ---
    ram_norm = obs['ram']
    print(f"\n3. RAM INPUT (Normalized)")
    print(f"Shape: {ram_norm.shape} (Total Stacked Bytes)")
    print(f"Type: {ram_norm.dtype}")
    print(f"Range: [{np.min(ram_norm):.2f}, {np.max(ram_norm):.2f}]")
    
    # Verify Content
    # RAM is stacked (4 frames). Let's look at the MOST RECENT frame (last 2048 bytes)
    # The stack is concatenated, so if stack=4, size is 8192. 
    # The LAST 2048 bytes are the current state.
    
    ram_size_per_frame = 2048
    stack_size = 4
    total_ram_size = ram_size_per_frame * stack_size
    
    # Slice the last frame
    current_ram_norm = ram_norm[-ram_size_per_frame:] 
    
    # Denormalize to see actual NES bytes (0-255)
    current_ram_bytes = (current_ram_norm * 255.0).astype(np.uint8)
    
    print(f"\n4. DECODED RAM VALUES (Current Frame)")
    
    # Known Addresses
    ADDR_LIVES = 0x51
    ADDR_X = 0x90
    ADDR_Y = 0x98
    ADDR_STAGE = 0x85
    ADDR_ENEMIES_LEFT = 0x82 # Example address (often used, need to check specific ROM)
    
    print(f"Player Lives (0x51): {current_ram_bytes[ADDR_LIVES]}")
    print(f"Player X (0x90): {current_ram_bytes[ADDR_X]}")
    print(f"Player Y (0x98): {current_ram_bytes[ADDR_Y]}")
    print(f"Stage (0x85): {current_ram_bytes[ADDR_STAGE]}")
    
    print("\n5. FULL MEMORY DUMP (Non-Zero values)")
    # Print first 100 non-zero bytes to show we are getting data
    count = 0
    for i in range(len(current_ram_bytes)):
        if current_ram_bytes[i] > 0:
            print(f"0x{i:03X}: {current_ram_bytes[i]}", end=" | ")
            count += 1
            if count > 20: 
                print("... (truncated)")
                break
    
    print("\n\nVERDICT:")
    if np.sum(current_ram_bytes) > 0:
        print("PASS: RAM contains data.")
    else:
        print("FAIL: RAM is empty (all zeros)!")
        
    env.close()

if __name__ == "__main__":
    inspect()
