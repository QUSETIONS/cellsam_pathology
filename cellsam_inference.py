import torch
import numpy as np
import os
import sys
from pathlib import Path

# Try to import CellSAM
try:
    from cellSAM import get_model, segment_cellular_image
except ImportError:
    print("Error: Could not import cellSAM. Ensure it is installed.")
    sys.exit(1)

class CellSAMInferencer:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        print(f"Initializing CellSAMInferencer on {self.device}...")
        self._load_model()

    def _load_model(self):
        try:
            print("Loading CellSAM model (this may verify/download weights)...")
            # This handles checking ~/.deepcell/models
            self.model = get_model()
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"\n[MODEL LOAD ERROR] {e}")
            print("="*60)
            print("Please ensure you have set the DEEPCELL_ACCESS_TOKEN environment variable.")
            print("Powershell example:")
            print('  $env:DEEPCELL_ACCESS_TOKEN="<your_token_here>"')
            print("Then run this script again.")
            print("="*60)
            # We don't exit here to allow the test script to handle it gracefully or show other things
            self.model = None

    def infer_on_patch(self, image_np):
        """
        Runs inference on a single patch.
        Args:
            image_np: Numpy array (H, W, 3) RGB, uint8
        Returns:
            mask: Numpy array (H, W) where each cell has a unique ID.
            embeddings: Image embeddings (optional, might be None)
        """
        if self.model is None:
            print("Model not loaded. Skipping inference.")
            return np.zeros(image_np.shape[:2], dtype=np.int32)

        try:
            # CellSAM expects standard numpy arrays
            # We can force mixed precision if supported, but typically 
            # inference is fast enough on 4060.
            # segment_cellular_image handles normalization internally if normalize=True
            
            print(f"Running inference on patch shape {image_np.shape}...")
            with torch.no_grad():
                # Note: 'fast=True' enables batched inference if supported/implemented
                mask, embeddings, boxes = segment_cellular_image(
                    image_np, 
                    self.model, 
                    device=self.device,
                    fast=True 
                )
            return mask
        except Exception as e:
            print(f"Inference error: {e}")
            return np.zeros(image_np.shape[:2], dtype=np.int32)
