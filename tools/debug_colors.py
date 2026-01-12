import os
import numpy as np
from PIL import Image

def analyze_colors():
    dir_path = "textures/tanks"
    if not os.path.exists(dir_path):
        print("Directory not found")
        return

    files = [f for f in os.listdir(dir_path) if f.endswith(".png")]
    files.sort()
    
    print(f"Found {len(files)} files.")
    
    for f in files[:20]:
        path = os.path.join(dir_path, f)
        img = Image.open(path).convert("RGBA")
        arr = np.array(img)
        
        # Get pixels where alpha > 0
        pixels = arr[arr[:,:,3] > 0]
        
        if len(pixels) == 0:
            print(f"{f}: Transparent")
            continue
            
        # Mean RGB
        mean_color = pixels[:, :3].mean(axis=0)
        print(f"{f}: RGB {mean_color.astype(int)}")

if __name__ == "__main__":
    analyze_colors()
