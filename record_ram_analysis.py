import gymnasium as gym
import numpy as np
from battle_city_env import BattleCityEnv
import time

def record_analysis():
    print("Initializing Environment...")
    env = BattleCityEnv(render_mode='human', use_vision=False) 
    env.reset()
    
    # Wait for game to settle (Start screen etc)
    print("Waiting for game start...")
    for _ in range(100):
        env.step(0) # NOOP

    prev_ram = np.array(env.raw_env.ram[:2048])
    
    log_file = "ram_dynamics_log.txt"
    
    with open(log_file, "w") as f:
        f.write("Frame | Action | RAM Changes (Address: Old vs New)\n")
        f.write("="*60 + "\n")
        
        # Define a choreography to isolate variables
        # 1. Idle (Baseline)
        # 2. Move Right (X should change)
        # 3. Move Down (Y should change)
        # 4. Fire (Bullet variables should appear)
        
        choreography = []
        choreography += [('WAIT', 0, 20)]   # Normalize
        choreography += [('RIGHT', 4, 30)]  # Move X
        choreography += [('WAIT', 0, 20)]
        choreography += [('DOWN', 2, 30)]   # Move Y
        choreography += [('WAIT', 0, 20)]
        choreography += [('FIRE', 5, 5)]    # Create Bullet
        choreography += [('WAIT', 0, 50)]   # Watch Bullet fly
        
        frame_idx = 0
        
        for label, action_idx, steps in choreography:
            f.write(f"--- START ACTION: {label} ---\n")
            print(f"Executing: {label}")
            
            for _ in range(steps):
                obs, reward, terminated, truncated, info = env.step(action_idx)
                # Render to ensure emulator runs at correct speed/integrity
                # env.render() 
                
                curr_ram = np.array(env.raw_env.ram[:2048])
                
                # Find differences
                diff_indices = np.where(curr_ram != prev_ram)[0]
                
                if len(diff_indices) > 0:
                    changes = []
                    for idx in diff_indices:
                        changes.append(f"0x{idx:04X}: {prev_ram[idx]:02X}->{curr_ram[idx]:02X}")
                    
                    change_str = ", ".join(changes)
                    f.write(f"{frame_idx:<5} | {label:<6} | {change_str}\n")
                
                prev_ram = curr_ram
                frame_idx += 1
                
                if terminated:
                    f.write("--- TERMINATED ---\n")
                    env.reset()
                    # Reset baseline
                    prev_ram = np.array(env.raw_env.ram[:2048])

    print(f"Analysis complete. Log saved to {log_file}")
    env.close()

if __name__ == "__main__":
    record_analysis()
