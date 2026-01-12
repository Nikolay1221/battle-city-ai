import os
import shutil
import numpy as np
from PIL import Image

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sort_tanks():
    base_dir = "textures/tanks"
    dirs = {
        "player": os.path.join(base_dir, "player"),
        "enemy": os.path.join(base_dir, "enemy"),
        "bonus": os.path.join(base_dir, "bonus"),
        "other": os.path.join(base_dir, "other")
    }
    
    for d in dirs.values():
        ensure_dir(d)
        
    # RESET STEP: Move all files from subfolders back to base_dir
    print("Resetting file locations...")
    for root, d_names, f_names in os.walk(base_dir):
        if root == base_dir: continue
        for f in f_names:
            if f.endswith(".png"):
                shutil.move(os.path.join(root, f), os.path.join(base_dir, f))
    
    # Re-list files
    files = [f for f in os.listdir(base_dir) if f.endswith(".png")]
    print(f"Sorting {len(files)} tanks...")
    
    counts = {k: 0 for k in dirs}
    
    for f in files:
        src = os.path.join(base_dir, f)
        try:
            img = Image.open(src).convert("RGBA")
            arr = np.array(img)
            pixels = arr[arr[:,:,3] > 10] # Non-transparent
            
            if len(pixels) == 0:
                target = "other"
            else:
                mean = pixels[:, :3].mean(axis=0)
                r, g, b = mean
                
                # Heuristics
                # Player: Yellow/Gold (High R, High G, Low B)
                is_yellow = (r > 130) and (g > 130) and (b < 110)
                
                # Bonus: Red/Purple/Pink
                # New Rule from tank_143 analysis: High Red, Low Green.
                # Tank 143: R=170, G=91, B=127.
                is_bonus = (r > 150) and (g < 120)
                
                if is_yellow:
                    target = "player"
                elif is_bonus:
                    target = "bonus"
                else:
                    # Enemy (Silver, Green, White)
                    target = "enemy"
            
            dst = os.path.join(dirs[target], f)
            shutil.move(src, dst)
            counts[target] += 1
            
        except Exception as e:
            print(f"Error {f}: {e}")
            
    print("Sort Complete!")
    for k, v in counts.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    sort_tanks()
