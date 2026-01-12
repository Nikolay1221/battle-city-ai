import gymnasium as gym
import numpy as np
from battle_city_env import BattleCityEnv
import time
import os

def monitor_ram_table():
    print("Initializing Environment...")
    env = BattleCityEnv(render_mode='human', use_vision=False) 
    env.reset()
    
    # Range to monitor (Specific Request: 0x90 - 0x9A)
    start_addr = 0x0090
    end_addr = 0x009A
    
    # Initialize Tracking Data
    ram_size = 2048
    current_ram = np.array(env.raw_env.ram[:ram_size], dtype=np.int16) # int16 to handle 'Previous' easily
    previous_ram = current_ram.copy()
    change_counts = np.zeros(ram_size, dtype=np.int32)
    
    log_file = "ram_table_log.txt"
    
    print(f"Monitoring RAM {hex(start_addr)} - {hex(end_addr)}...")
    print(f"Saving formatted snapshots to {log_file}...")
    
    with open(log_file, "w") as f:
        # Run for a set number of frames
        for frame_idx in range(600):
            # NOOP to let game play
            env.step(0) 
            env.render()
            
            # Update RAM
            new_ram = np.array(env.raw_env.ram[:ram_size], dtype=np.int16)
            
            # Detect Changes
            diff_indices = np.where(new_ram != current_ram)[0]
            for idx in diff_indices:
                change_counts[idx] += 1
            
            # Update history
            previous_ram = current_ram.copy()
            current_ram = new_ram.copy()
            
            # Log Snapshot every 60 frames (1 sec) OR if significant activity in our range
            # To avoid spamming, let's just log every 1 second (60 frames)
            if frame_idx % 60 == 0:
                f.write(f"\n=== SNAPSHOT AT FRAME {frame_idx} ===\n")
                f.write(f"{'Addr.':<6} | {'Value':<5} | {'Previous':<8} | {'Changes':<7}\n")
                f.write("-" * 35 + "\n")
                
                for addr in range(start_addr, end_addr + 1):
                    val = current_ram[addr]
                    prev = previous_ram[addr]
                    changes = change_counts[addr]
                    
                    # Formatting
                    # Addr: Hex (0040)
                    # Value: Decimal (Unsigned or Signed? Image had -3. Let's do Unsigned (Signed))
                    
                    # Python bytes are 0-255. 
                    # If user wants signed:
                    val_signed = val if val <= 127 else val - 256
                    prev_signed = prev if prev <= 127 else prev - 256
                    
                    # We will show Unsigned (Signed) for clarity
                    # e.g. "253 (-3)"
                    
                    f.write(f"{addr:04X}   | {val:<5} | {prev:<8} | {changes:<7}\n")
                    
                f.write("-" * 35 + "\n")

    print(f"Done. Formatted log saved to {log_file}")
    env.close()

if __name__ == "__main__":
    monitor_ram_table()
