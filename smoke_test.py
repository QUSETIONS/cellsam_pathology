import torch
import os
import numpy as np
import sys

print(f"Python version: {sys.version}")

try:
    from cellSAM import segment_cellular_image, get_model
    print("Import successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    # Don't exit yet, let's see what else works
    
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

# Create a dummy image
img = np.zeros((256, 256, 3), dtype=np.uint8)
img[50:100, 50:100] = 255 # White square

try:
    print("Attempting to load model (this may fail if weights are missing)...")
    # This will trigger download if not present, and fail if no token
    # We catch the specific auth error if possible, or generic
    model = get_model() 
    print("Model loaded!")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running inference on {device}...")
    mask, _, _ = segment_cellular_image(img, model, device=device)
    print(f"Inference successful. Mask shape: {mask.shape}")
    
    # Save output to prove it ran
    from PIL import Image
    Image.fromarray((mask * 255).astype(np.uint8)).save("smoke_test_output.png")
    print("Output saved to smoke_test_output.png")

except Exception as e:
    print(f"\n[EXPECTED IF NO TOKEN] Model load/inference failed.")
    print(f"Error details: {e}")
    print("\nTo fix this for real usage:")
    print("1. Register at users.deepcell.org")
    print("2. Set DEEPCELL_ACCESS_TOKEN environment variable.")
    print("3. Run the script again to download weights.")
    print("4. Or manually place 'cellsam_general.pt' in ~/.deepcell/models/cellsam_v1.2/")
