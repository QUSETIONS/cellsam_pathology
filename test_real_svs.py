import matplotlib.pyplot as plt
import os
from wsi_handler import WSILoader

# Use the specific file provided in context
SVS_PATH = r"D:\BaiduNetdiskDownload\202550016.21.svs"

def run_test():
    print(f"Testing WSI access for: {SVS_PATH}")
    
    if not os.path.exists(SVS_PATH):
        print(f"Error: File not found at {SVS_PATH}")
        return

    try:
        loader = WSILoader(SVS_PATH)
        loader.get_info()
        
        # Define a center point. 
        # For a robust test, pick the center of the slide
        w, h = loader.slide.dimensions
        cx, cy = w // 2, h // 2
        
        print(f"Reading center patch at ({cx}, {cy})...")
        
        # 1. Read Level 0 (High Res)
        # Using a smaller size for visual verification to keep file size sane, but 1024 is requested.
        # We'll use 512 for display to save space, but 1024 for the test logic.
        patch_l0 = loader.read_region_as_tensor(cx, cy, 0, (1024, 1024))
        print(f"Level 0 Patch shape: {patch_l0.shape}")
        
        # 2. Read Lower Level (e.g., Level 2)
        # Check if level 2 exists
        target_level = 2 if loader.slide.level_count > 2 else loader.slide.level_count - 1
        print(f"Reading Level {target_level} patch...")
        patch_low = loader.read_region_as_tensor(cx, cy, target_level, (1024, 1024))
        print(f"Level {target_level} Patch shape: {patch_low.shape}")
        
        # Visualize
        print("Saving visualization...")
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        ax[0].imshow(patch_l0)
        ax[0].set_title("Level 0 (Center)")
        ax[0].axis("off")
        
        ax[1].imshow(patch_low)
        ax[1].set_title(f"Level {target_level} (Center)")
        ax[1].axis("off")
        
        plt.tight_layout()
        plt.savefig("wsi_test_result.png")
        print("Successfully saved 'wsi_test_result.png'")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
