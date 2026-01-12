import os
import cv2
import numpy as np
import glob
import shutil

def main():
    source_dir = "textures/tanks/enemy"
    dest_dir = "templates/enemies"
    
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    
    files = sorted(glob.glob(os.path.join(source_dir, "*.png")))
    print(f"Found {len(files)} candidate sprites.")
    
    selected_templates = []
    
    saved_count = 0
    
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # Resize for comparison (to ignore minor pixel shifts/upscaling artifacts)
        img_small = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
        
        is_unique = True
        min_diff = 999999
        
        for temp_small in selected_templates:
            diff = cv2.absdiff(img_small, temp_small)
            # Sum of absolute differences
            diff_score = np.sum(diff)
            if diff_score < min_diff: min_diff = diff_score
            
            # If very similar (avg pixel diff < 30 roughly)
            # 16*16*30 = 7680
            # Let's try 10000 (~40 intensity diff per pixel avg)
            if diff_score < 10000: # Very Aggressive threshold
                is_unique = False
                break
        
        if is_unique:
            selected_templates.append(img_small) # Store small for comparison
            # Save RESIZED 16x16 (Correct scale for NESEnv)
            out_name = os.path.join(dest_dir, f"enemy_{saved_count:02d}.png")
            cv2.imwrite(out_name, img_small)
            saved_count += 1
            print(f"Selected {f} (MinDiff: {min_diff})")
        else:
            pass
            # print(f"Skipped {f} (MinDiff: {min_diff})")
            
    print(f"Reduced {len(files)} sprites to {saved_count} unique templates.")

if __name__ == "__main__":
    main()
