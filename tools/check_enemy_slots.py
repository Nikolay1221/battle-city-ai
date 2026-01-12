import sys
import os
# Ensure we can import battle_city_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from battle_city_env import BattleCityEnv
import time
import numpy as np

def main():
    print("Initializing Environment...")
    env = BattleCityEnv(render_mode='human', use_vision=False)
    env.reset()
    
    # Potential Addresses
    # Player X: 0x90.  Slots 1-5? -> 0x91...
    # Player Y: 0x98.  Slots 1-5? -> 0x99...
    
    print("Watching RAM Slots 0x90..0x9F")
    print("Columns: [X0 P] [X1] [X2] [X3] [X4] | [Y0 P] [Y1] [Y2] [Y3] [Y4]")
    
    try:
        while True:
            # Step to keep game running
            action = env.action_space.sample() # Random action to make things happen? 
            # Or NOOP to just watch
            obs, _, _, _, _ = env.step(0)
            
            ram = env.raw_env.ram
            
            # X Coords (0x90 base)
            xs = [ram[0x90 + i] for i in range(5)]
            # Y Coords (0x98 base)
            ys = [ram[0x98 + i] for i in range(5)]
            
            # Format
            x_str = " ".join([f"{x:3d}" for x in xs])
            y_str = " ".join([f"{y:3d}" for y in ys])
            
            print(f"\r X: [{x_str}] | Y: [{y_str}]", end="")
            
            env.render()
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
