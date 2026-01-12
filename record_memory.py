import gymnasium as gym
import numpy as np
from battle_city_env import BattleCityEnv
import time

def dump_memory():
    print("Initializing Environment...")
    # Initialize env (Vision not needed for RAM dump)
    env = BattleCityEnv(render_mode='human', use_vision=False) 
    env.reset()
    
    print("Running warmup steps to reach active gameplay...")
    # Run some steps to get past menus or just to get a "live" state
    for i in range(200):
        # Action 0 = NOOP usually, or random
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.01) # Slow down to watch
        
        if terminated or truncated:
            env.reset()

    print("Capturing RAM...")
    # Access Raw RAM from NES emulator
    # The NES has 2KB of internal RAM (0x0000 - 0x07FF)
    # Plus cartridge RAM if any. nes_py usually gives the full addressable memory or system RAM.
    # Typically 'env.raw_env.ram' is a bounded array.
    
    ram_data = env.raw_env.ram
    
    # We'll grab the first 2048 bytes (Standard NES RAM)
    # Usually getting everything is safer to see what we have.
    print(f"Total RAM accessible: {len(ram_data)} bytes")
    
    # 1. Save as Binary
    with open("ram_dump.bin", "wb") as f:
        # Convert to bytes
        # ram_data is usually a numpy array or similar-like buffer
        f.write(bytearray(ram_data))
    print("Saved 'ram_dump.bin'")

    # 2. Save as Hex Text (Readable)
    with open("ram_dump.txt", "w") as f:
        f.write(f"Address | 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F | ASCII\n")
        f.write("-" * 78 + "\n")
        
        for i in range(0, 2048, 16): # Just show first 2KB (Main RAM)
            chunk = ram_data[i:i+16]
            hex_str = " ".join([f"{b:02X}" for b in chunk])
            
            # Simple ASCII repr
            ascii_str = ""
            for b in chunk:
                if 32 <= b <= 126:
                    ascii_str += chr(b)
                else:
                    ascii_str += "."
            
            f.write(f"0x{i:04X}  | {hex_str:<47} | {ascii_str}\n")
            
    print("Saved 'ram_dump.txt' (First 2KB)")
    
    env.close()

if __name__ == "__main__":
    dump_memory()
