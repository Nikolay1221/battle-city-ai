import numpy as np
from PIL import Image
import os
import sys

# Try importing scipy for faster labeling, else fallback to custom
try:
    from scipy.ndimage import label, find_objects
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_textures(image_path, output_dir="textures"):
    print(f"Loading {image_path}...")
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Convert to numpy
    arr = np.array(img)
    
    # Extract Alpha Channel
    alpha = arr[:, :, 3]
    
    # Create Binary Mask (Threshold at 10 to ignore near-invisible noise)
    mask = alpha > 10
    
    print(f"Image Size: {img.size}")
    print(f"Scipy Available: {HAS_SCIPY}")

    ensure_dir(output_dir)

    objects_slices = []

    if HAS_SCIPY:
        print("Using Scipy for blob detection...")
        labeled_array, num_features = label(mask)
        objects_slices = find_objects(labeled_array)
        print(f"Detected {num_features} objects.")
    else:
        print("Scipy not found. Using custom Flood Fill (this might take a moment)...")
        # Custom basic connected components (slow python loop, but works for sprite sheets)
        # Optimized: Scan for True pixels, run iterative fill
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        
        # Helper for non-recursive flood fill
        def get_blob_bbox(start_y, start_x):
            min_y, max_y = start_y, start_y
            min_x, max_x = start_x, start_x
            
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            
            while stack:
                cy, cx = stack.pop()
                
                # Update BBox
                if cy < min_y: min_y = cy
                if cy > max_y: max_y = cy
                if cx < min_x: min_x = cx
                if cx > max_x: max_x = cx
                
                # Neighbors (4-connectivity)
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                            
            return (slice(min_y, max_y+1), slice(min_x, max_x+1))

        for y in range(h):
            for x in range(w):
                if mask[y, x] and not visited[y, x]:
                    bbox = get_blob_bbox(y, x)
                    objects_slices.append(bbox)
        
        print(f"Detected {len(objects_slices)} objects.")

    print(f"Extracting to '{output_dir}/'...")
    
    count = 0
    for i, sl in enumerate(objects_slices):
        if sl is None: continue
        
        # Get crop
        # sl is tuple of slices (slice(y_start, y_end), slice(x_start, x_end))
        dy, dx = sl
        
        # Check size (ignore single pixels/noise)
        h_obj = dy.stop - dy.start
        w_obj = dx.stop - dx.start
        if h_obj < 2 or w_obj < 2:
            continue
            
        sprite = img.crop((dx.start, dy.start, dx.stop, dy.stop))
        
        # Save
        filename = f"texture_{count:03d}.png"
        save_path = os.path.join(output_dir, filename)
        sprite.save(save_path)
        
        # Debug print every 50
        if count % 50 == 0:
             print(f"Saved {filename} ({w_obj}x{h_obj})")
        
        count += 1
        
    print(f"\nDone! Extracted {count} textures.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        src = "image.png"
        
    extract_textures(src)
