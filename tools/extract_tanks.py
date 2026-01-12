import numpy as np
from PIL import Image
import os
import sys
import math

# Try importing scipy for faster labeling, else fallback to custom
try:
    from scipy.ndimage import label, find_objects
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_tanks(image_path, output_dir="textures/tanks"):
    print(f"Loading {image_path}...")
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    arr = np.array(img)
    alpha = arr[:, :, 3]
    mask = alpha > 10 # Threshold
    
    ensure_dir(output_dir)

    objects_slices = []

    if HAS_SCIPY:
        print("Using Scipy for blob detection...")
        labeled_array, num_features = label(mask)
        objects_slices = find_objects(labeled_array)
    else:
        print("Using custom Flood Fill...")
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        
        def get_blob_bbox(start_y, start_x):
            min_y, max_y = start_y, start_y
            min_x, max_x = start_x, start_x
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            while stack:
                cy, cx = stack.pop()
                if cy < min_y: min_y = cy
                if cy > max_y: max_y = cy
                if cx < min_x: min_x = cx
                if cx > max_x: max_x = cx
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
                    objects_slices.append(get_blob_bbox(y, x))

    print(f"Found {len(objects_slices)} objects total. Analyzing sizes...")
    
    # Debug: Print common sizes
    sizes = []
    for i, sl in enumerate(objects_slices):
        if sl is None: continue
        dy, dx = sl
        h = dy.stop - dy.start
        w = dx.stop - dx.start
        sizes.append((w, h))
        
    # Print top 10 most common sizes
    from collections import Counter
    common = Counter(sizes).most_common(20)
    print("Most common blob sizes (Width x Height):")
    for size, count in common:
        print(f"  {size[0]}x{size[1]}: {count} occurrences")

    tanks = []
    
    # Filter for Tanks (ADJUSTED FINAL)
    # Common sizes found: 52x52, 60x52, 52x60.
    # We set a broad range to catch all of them.
    MIN_SIZE = 40
    MAX_SIZE = 75
    
    for i, sl in enumerate(objects_slices):
        if sl is None: continue
        dy, dx = sl
        h_obj = dy.stop - dy.start
        w_obj = dx.stop - dx.start
        
        if MIN_SIZE <= h_obj <= MAX_SIZE and MIN_SIZE <= w_obj <= MAX_SIZE:
             sprite = img.crop((dx.start, dy.start, dx.stop, dy.stop))
             tanks.append(sprite)
            
    if not tanks:
        print("No tanks found! Check size thresholds.")
        return

    print(f"Filtered down to {len(tanks)} tank candidates.")
    
    # Save Individual Files
    for i, sprite in enumerate(tanks):
        filename = f"tank_{i:03d}.png"
        sprite.save(os.path.join(output_dir, filename))
        
    # Generate Preview Sheet
    # Grid: approx square
    grid_w = int(math.ceil(math.sqrt(len(tanks))))
    grid_h = int(math.ceil(len(tanks) / grid_w))
    tile_size = 20 # 16 + padding
    
    preview_img = Image.new('RGBA', (grid_w * tile_size, grid_h * tile_size), (0, 0, 0, 0))
    
    for idx, sprite in enumerate(tanks):
        row = idx // grid_w
        col = idx % grid_w
        x = col * tile_size + (tile_size - sprite.width)//2
        y = row * tile_size + (tile_size - sprite.height)//2
        preview_img.paste(sprite, (x, y))
        
    preview_path = "tanks_preview.png"
    preview_img.save(preview_path)
    print(f"Saved preview to {preview_path}")

if __name__ == "__main__":
    src = "image.png"
    if len(sys.argv) > 1: src = sys.argv[1]
    extract_tanks(src)
