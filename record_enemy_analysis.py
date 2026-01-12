import gymnasium as gym
import numpy as np
from battle_city_env import BattleCityEnv
import time

def record_enemy_analysis():
    print("Initializing Environment for Enemy Tracking...")
    env = BattleCityEnv(render_mode='human', use_vision=False) 
    env.reset()
    
    log_file = "enemy_ram_log.txt"
    
    # Potential Enemy Addresses (from hypothesis + range)
    # X: 0x56..0x59 ?
    # Y: 0x5A..0x5D ?
    # Let's monitor a wider range just in case: 0x40 to 0x70
    start_addr = 0x40
    end_addr = 0x70
    
    print(f"Monitoring RAM {hex(start_addr)} - {hex(end_addr)} for 600 frames...")
    
    with open(log_file, "w") as f:
        f.write("Frame | Monitor Range (0x40 - 0x70)\n")
        f.write("-" * 100 + "\n")
        
        # Header for columns
        header = "      "
        for i in range(start_addr, end_addr + 1):
             header += f"{i:02X} "
        f.write(header + "\n")
        
        for i in range(600):
            # NOOP - Let enemies spawn and move
            env.step(0) 
            env.render()
            
            ram = env.raw_env.ram
            
            # Extract chunk
            chunk = ram[start_addr:end_addr+1]
            
            # Format as string
            vals = " ".join([f"{b:02X}" for b in chunk])
            f.write(f"{i:<5} | {vals}\n")
            
            # Sleep slightly to let user watch if they want (fast forward)
            # time.sleep(0.005) 
            
    print(f"Done. Log saved to {log_file}")
    env.close()

if __name__ == "__main__":
    record_enemy_analysis()
