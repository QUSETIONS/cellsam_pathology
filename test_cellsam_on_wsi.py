import matplotlib.pyplot as plt
import numpy as np
import os
import time
from wsi_handler import WSILoader
from cellsam_inference import CellSAMInferencer
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries

SVS_PATH = r"D:\BaiduNetdiskDownload\202550016.21.svs"
OUTPUT_FILENAME = "cellsam_wsi_results.png"

def run_end_to_end_test():
    print("Starting End-to-End WSI + CellSAM Test...")
    
    # 1. Initialize WSI Loader
    loader = WSILoader(SVS_PATH)
    loader.get_info()
    
    # 2. Initialize Model
    inferencer = CellSAMInferencer()
    
    # 3. Detect Tissue & Select Patches
    print("Detecting tissue regions...")
    points, thumb = loader.detect_tissue_regions()
    
    if not points:
        print("No tissue detected. Using center.")
        w, h = loader.slide.dimensions
        points = [(w//2, h//2)]
    
    # Select up to 3 points
    # If we have enough, let's try to make them distinct.
    # points are sorted by area size from detect_tissue_regions
    selected_points = points[:3]
    print(f"Selected inference points (Level 0): {selected_points}")
    
    # 4. Process Patches
    results = []
    patch_size = 1024
    
    for idx, (cx, cy) in enumerate(selected_points):
        print(f"\n--- Processing Patch {idx+1}/{len(selected_points)} at ({cx}, {cy}) ---")
        
        # Read Patch
        t0 = time.time()
        patch = loader.read_region_as_tensor(cx, cy, 0, (patch_size, patch_size))
        read_time = time.time() - t0
        
        if patch is None:
            print("Failed to read patch.")
            continue
            
        # Inference
        t1 = time.time()
        if inferencer.model:
            mask = inferencer.infer_on_patch(patch)
            cell_count = len(np.unique(mask)) - 1 # 0 is background
        else:
            mask = np.zeros((patch_size, patch_size), dtype=np.int32)
            cell_count = 0
        infer_time = time.time() - t1
        
        print(f"Patch Read: {read_time:.3f}s | Inference: {infer_time:.3f}s | Cells Detected: {cell_count}")
        
        results.append({
            'patch': patch,
            'mask': mask,
            'point': (cx, cy),
            'cells': cell_count
        })

    # 5. Visualization
    print("\nGenerating result image...")
    rows = len(results)
    if rows == 0:
        print("No results to visualize.")
        return

    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    # If only 1 row, axes is 1D array
    if rows == 1:
        axes = np.expand_numpy(axes, axis=0) # Make it 2D array [1, 3] -> No, expand_dims
        axes = axes.reshape(1, -1)

    # Prepare thumbnail with markers
    thumb_np = np.array(thumb)
    w_l0, h_l0 = loader.slide.dimensions
    scale_x = thumb.size[0] / w_l0
    scale_y = thumb.size[1] / h_l0

    import matplotlib.patches as patches
    
    # We will just plot the thumbnail once in a separate figure or 
    # make the layout: Left Column (Thumb), Right Column (Patches)
    # Actually requested: Thumb (marked), then 3 groups of (Original vs Mask)
    # Let's do a Grid:
    # Top: Thumbnail
    # Bottom: Row per patch (Original, Overlay, Mask Only)
    
    plt.close(fig) # Reset
    
    fig = plt.figure(figsize=(20, 5 + 5 * rows), layout="constrained")
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 3 * rows])
    
    # Top: Thumbnail
    ax_thumb = subfigs[0].subplots(1, 1)
    ax_thumb.imshow(thumb)
    ax_thumb.set_title(f"Whole Slide Thumbnail with {rows} Sample Regions")
    
    colors = ['r', 'g', 'b', 'y', 'm']
    
    for i, res in enumerate(results):
        cx, cy = res['point']
        tx = cx * scale_x
        ty = cy * scale_y
        
        # Draw on thumbnail
        rect = patches.Rectangle(
            (tx - (patch_size*scale_x)/2, ty - (patch_size*scale_y)/2),
            patch_size*scale_x * 5, # Exaggerate size for visibility
            patch_size*scale_y * 5,
            linewidth=2, edgecolor=colors[i % len(colors)], facecolor='none'
        )
        ax_thumb.add_patch(rect)
        ax_thumb.text(tx, ty, str(i+1), color=colors[i % len(colors)], fontsize=12, fontweight='bold')

    # Bottom: Patches
    axs_patches = subfigs[1].subplots(rows, 2)
    if rows == 1: axs_patches = axs_patches.reshape(1, -1)
    
    for i, res in enumerate(results):
        patch = res['patch']
        mask = res['mask']
        count = res['cells']
        
        # Original
        axs_patches[i, 0].imshow(patch)
        axs_patches[i, 0].set_title(f"Region {i+1} (Cells: {count})", color=colors[i % len(colors)])
        axs_patches[i, 0].axis('off')
        
        # Overlay
        # label2rgb creates a float image [0,1], convert back to uint8 for consistency if needed
        # or mark_boundaries
        if count > 0:
            overlay = mark_boundaries(patch, mask, color=(0, 1, 0), mode='thick')
        else:
            overlay = patch
            
        axs_patches[i, 1].imshow(overlay)
        axs_patches[i, 1].set_title(f"Segmentation Overlay")
        axs_patches[i, 1].axis('off')

    plt.savefig(OUTPUT_FILENAME)
    print(f"Visualization saved to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    run_end_to_end_test()
