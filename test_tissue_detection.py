import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from wsi_handler import WSILoader
from PIL import Image

# Use the specific file provided in context
SVS_PATH = r"D:\BaiduNetdiskDownload\202550016.21.svs"

def detect_tissue(thumb_img):
    """
    Simple tissue detection on a thumbnail image.
    Returns:
        mask: Boolean mask of tissue
        (cx, cy): Centroid of the largest tissue region (in thumbnail coordinates)
        bbox: (min_row, min_col, max_row, max_col) of the largest region
    """
    # Convert to numpy array
    arr = np.array(thumb_img)
    
    # Simple thresholding for H&E: Background is usually bright/white
    # Tissue is darker. We can use the green channel or just average.
    # Gray scale conversion
    gray = np.mean(arr, axis=2)
    
    # Thresholding: background is bright (> 210 approx)
    # Inverting so tissue is True (1), background is False (0)
    binary = gray < 215
    
    # Clean up noise (optional, but good for "salt and pepper" noise)
    # Using scipy.ndimage for connected components if available, 
    # or just use simple numpy tricks if we want to avoid deps.
    # Since we have scipy installed:
    from scipy.ndimage import label, find_objects
    
    labeled, n_components = label(binary)
    
    if n_components == 0:
        return None, None, None
        
    # Find largest component
    largest_slice = None
    max_area = 0
    largest_label = 0
    
    # Basic region props
    slices = find_objects(labeled)
    for i, sl in enumerate(slices):
        if sl is None: continue
        # Approximate area by bounding box size to be fast, 
        # or count pixels for accuracy. Let's count pixels.
        # extracting the region from binary mask
        component_mask = (labeled == (i + 1))
        area = np.sum(component_mask)
        
        if area > max_area:
            max_area = area
            largest_slice = sl
            largest_label = i + 1
            
    if largest_slice is None:
        return None, None, None
        
    # Calculate centroid of the largest region
    # Center of the bounding box is a good enough approximation for patch reading
    # But let's try to be slightly more precise: Center of Mass
    coords = np.argwhere(labeled == largest_label)
    cy = int(np.mean(coords[:, 0]))
    cx = int(np.mean(coords[:, 1]))
    
    # Bounding box: min_row, min_col, max_row, max_col
    # from slice object: start, stop
    min_row, max_row = largest_slice[0].start, largest_slice[0].stop
    min_col, max_col = largest_slice[1].start, largest_slice[1].stop
    
    return binary, (cx, cy), (min_row, min_col, max_row, max_col)

def run_tissue_test():
    print(f"Testing Tissue Detection on: {SVS_PATH}")
    
    if not os.path.exists(SVS_PATH):
        print(f"Error: File not found at {SVS_PATH}")
        return

    try:
        loader = WSILoader(SVS_PATH)
        loader.get_info()
        
        # 1. Generate Thumbnail
        print("Generating thumbnail...")
        # Get a thumbnail of max size 1024x1024
        thumb = loader.slide.get_thumbnail((1024, 1024))
        thumb.save("wsi_thumbnail.png")
        print("Saved 'wsi_thumbnail.png'")
        
        # 2. Detect Tissue
        print("Detecting tissue...")
        mask, centroid, bbox = detect_tissue(thumb)
        
        if centroid is None:
            print("WARNING: No tissue detected! Defaulting to center.")
            w, h = loader.slide.dimensions
            cx_l0, cy_l0 = w // 2, h // 2
            # Dummy bbox for plotting
            bbox = (0, 0, thumb.size[1], thumb.size[0])
            thumb_cx, thumb_cy = thumb.size[0]//2, thumb.size[1]//2
        else:
            thumb_cx, thumb_cy = centroid
            print(f"Tissue detected at Thumbnail coords: ({thumb_cx}, {thumb_cy})")
            
            # Map to Level 0
            w_thumb, h_thumb = thumb.size
            w_l0, h_l0 = loader.slide.dimensions
            
            scale_x = w_l0 / w_thumb
            scale_y = h_l0 / h_thumb
            
            cx_l0 = int(thumb_cx * scale_x)
            cy_l0 = int(thumb_cy * scale_y)
            
            print(f"Mapped to Level 0 coords: ({cx_l0}, {cy_l0})")

        # 3. Read Patches
        patch_size = 1024
        print(f"Reading Level 0 patch at ({cx_l0}, {cy_l0})...")
        patch_l0 = loader.read_region_as_tensor(cx_l0, cy_l0, 0, (patch_size, patch_size))
        
        target_level = 2 if loader.slide.level_count > 2 else loader.slide.level_count - 1
        print(f"Reading Level {target_level} patch at ({cx_l0}, {cy_l0})...")
        patch_low = loader.read_region_as_tensor(cx_l0, cy_l0, target_level, (patch_size, patch_size))
        
        # 4. Visualization
        print("Creating visualization...")
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Thumbnail with annotations
        ax[0].imshow(thumb)
        ax[0].set_title(f"Thumbnail ({thumb.size[0]}x{thumb.size[1]})")
        
        # Draw bounding box of detected tissue
        if bbox:
            # bbox is (min_row, min_col, max_row, max_col) -> (y1, x1, y2, x2)
            # Rect takes (x, y), w, h
            y1, x1, y2, x2 = bbox
            rect_tissue = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='g', facecolor='none')
            ax[0].add_patch(rect_tissue)
            
        # Draw reading point
        ax[0].plot(thumb_cx, thumb_cy, 'r+', markersize=15, markeredgewidth=2)
        # Draw approximate field of view of the patch (might be tiny on thumbnail)
        # 1024 pixels at Level 0 -> ? pixels on thumbnail
        patch_w_thumb = patch_size / scale_x
        patch_h_thumb = patch_size / scale_y
        rect_fov = patches.Rectangle(
            (thumb_cx - patch_w_thumb/2, thumb_cy - patch_h_thumb/2), 
            patch_w_thumb, patch_h_thumb, 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax[0].add_patch(rect_fov)
        
        # Plot 2: Level 0
        if patch_l0 is not None:
            ax[1].imshow(patch_l0)
            ax[1].set_title(f"Level 0 (High Res)\nCenter: ({cx_l0}, {cy_l0})")
        else:
            ax[1].text(0.5, 0.5, "Read Failed", ha='center', va='center')
        ax[1].axis("off")
        
        # Plot 3: Low Level
        if patch_low is not None:
            ax[2].imshow(patch_low)
            ax[2].set_title(f"Level {target_level} (Context)")
        else:
            ax[2].text(0.5, 0.5, "Read Failed", ha='center', va='center')
        ax[2].axis("off")
        
        plt.tight_layout()
        save_path = "wsi_tissue_test.png"
        plt.savefig(save_path)
        print(f"Successfully saved '{save_path}'")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_tissue_test()
