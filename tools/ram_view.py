import gymnasium as gym
import time
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battle_city_env import BattleCityEnv

class RamViewer:
    def __init__(self):
        print("Initializing Battle City Environment...")
        self.env = BattleCityEnv(render_mode='human', use_vision=False)
        self.env.reset()
        self.running = True
        
        # Known Addresses from battle_city_env.py
        self.ADDR_MAP = {
            "LIVES": 0x51,
            "STAGE": 0x85,
            "PLAYER_X": 0x90,
            "PLAYER_Y": 0x98,
            "BONUS": 0x62,
            "STATE": 0x92
        }

    def run(self):
        print("\n--- BATTLE CITY KNOW-IT-ALL ---")
        print("Monitoring known memory addresses...")
        print("Press Ctrl+C to stop.")
        
        try:
            while self.running:
                # Advance game
                self.env.step(0) 
                self.env.render()
                
                # Read RAM
                ram = self.env.raw_env.ram
                
                # Print Stats (Overwrite same line for clean look)
                output = []
                for name, addr in self.ADDR_MAP.items():
                    val = ram[addr]
                    output.append(f"{name}: {val}")
                
                # Monitor potential object arrays
                # Player is known at 0x90(X) / 0x98(Y) ?  Wait, code said 0x90/0x98.
                # Let's verify 0x90..0x9F (Possible X array) and 0x40..0x60
                
                ranges = [
                    ("X_Coords", 0x90, 8), # 0x90, 91, 92...
                    ("Y_Coords", 0x98, 8), # 0x98... ? Need to check stride
                    ("Possible_Y", 0x30, 16),
                    ("Lives?", 0x51, 1)
                ]
                
                print("\nXXX RAM WATCHER XXX")
                lines = []
                
                for label, start_addr, count in ranges:
                    vals = [f"{ram[start_addr+i]:02X}" for i in range(count)]
                    lines.append(f"{label}: {' '.join(vals)}")
                    
                print("\n".join(lines))
                print(f"\r{' | '.join(output)}", end="")
                
                time.sleep(1/60.0)
                
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            self.env.close()

if __name__ == "__main__":
    viewer = RamViewer()
    viewer.run()
