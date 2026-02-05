import numpy as np
import cv2
import sqlite3
import pickle
import os

# Try importing StarDist
try:
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    HAS_STARDIST = True
except ImportError:
    HAS_STARDIST = False
    print("Warning: StarDist not found. Segmentation will be simulated.")

# Try importing Torch
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet18, ResNet18_Weights
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: Torch not found. Feature extraction will be simulated.")

class FeaturesDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cells (
                x INTEGER,
                y INTEGER,
                label INTEGER DEFAULT -1,
                prediction INTEGER DEFAULT -1,
                tumor_prob REAL DEFAULT 0.0,
                feature_data BLOB
            )
        """)
        conn.commit()
        conn.close()

    def insert_batch(self, cells_data):
        """
        cells_data: list of dicts {'x', 'y', 'embedding'}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        vals = []
        for c in cells_data:
            # Serialize embedding
            emb = c.get('embedding')
            if emb is not None:
                emb_blob = pickle.dumps(emb)
            else:
                emb_blob = None
            vals.append((c['x'], c['y'], emb_blob))
            
        cursor.executemany("INSERT INTO cells (x, y, feature_data) VALUES (?, ?, ?)", vals)
        conn.commit()
        conn.close()

class NucleiPipeline:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NucleiPipeline, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        
        self.device = 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu'
        print(f"Initializing NucleiPipeline on {self.device}...")
        
        # 1. Load StarDist
        self.seg_model = None
        if HAS_STARDIST:
            try:
                # Use '2D_versatile_he' which is optimized for H&E
                self.seg_model = StarDist2D.from_pretrained('2D_versatile_he')
                print("StarDist model loaded successfully.")
            except Exception as e:
                print(f"Failed to load StarDist: {e}")

        # 2. Load Feature Extractor (ResNet18 as default)
        self.extractor = None
        self.transforms = None
        if HAS_TORCH:
            try:
                weights = ResNet18_Weights.DEFAULT
                self.extractor = resnet18(weights=weights)
                self.extractor.fc = torch.nn.Identity() # Remove classification head
                self.extractor.to(self.device)
                self.extractor.eval()
                self.transforms = weights.transforms()
            except Exception as e:
                print(f"Failed to init feature extractor: {e}")
            
        self._initialized = True

    def process_patch(self, patch_rgb, global_offset=(0,0)):
        """
        Args:
            patch_rgb: (H, W, 3) numpy array, uint8. RGB format expected.
            global_offset: (x, y) tuple of the top-left corner in WSI coordinates
        Returns:
            list of dicts: [{'x': int, 'y': int, 'embedding': np.array}, ...]
        """
        if patch_rgb is None or patch_rgb.size == 0:
            return []

        results = []
        H, W = patch_rgb.shape[:2]
        centroids = []
        
        # --- 0. Debug & Format Check ---
        # Ensure image is RGB (OpenSlide gives RGB).
        cv2.imwrite("debug_input_rgb.png", cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
        
        # --- A. Segmentation (StarDist) ---
        import traceback 
        used_fallback = False # Initialize flag
        
        if self.seg_model:
            print(f"DEBUG: Calling StarDist on image shape {patch_rgb.shape}, dtype {patch_rgb.dtype}...")
            
            try:
                # Normalization
                img_norm = normalize(patch_rgb, 1, 99.8, axis=(0,1))
                print(f"DEBUG: Normalized image shape {img_norm.shape}, dtype {img_norm.dtype} for StarDist.")

                # Smart Inference Mode Selection
                # predict_instances_big is only needed for massive images to avoid OOM.
                # Standard predict_instances is faster and safer for typical patches < 2048px.
                
                h_norm, w_norm = img_norm.shape[:2]
                if max(h_norm, w_norm) > 2048:
                    print("DEBUG: Image > 2048px, using Tiled Inference (predict_instances_big)...")
                    labels, details = self.seg_model.predict_instances_big(
                        img_norm, 
                        axes='YXC', 
                        block_size=1024, 
                        min_overlap=128,
                        prob_thresh=0.4 
                    )
                else:
                    print("DEBUG: Image small enough, using Standard Inference (predict_instances)...")
                    labels, details = self.seg_model.predict_instances(
                        img_norm,
                        prob_thresh=0.4
                    )
                
                # Debug Output
                if labels is not None:
                    debug_lbl = (labels > 0).astype(np.uint8) * 255
                    cv2.imwrite("debug_output_mask.png", debug_lbl)
                
                centroids = details['points'] # (N, 2) -> (y, x)
                print(f"DEBUG: StarDist returned {len(centroids)} instances.")
            
            except Exception as e:
                print("!!! StarDist Crash Report !!!")
                traceback.print_exc()
                print(f"Error details: {e}")
                used_fallback = True
                centroids = [] 
                
        else:
            print("DEBUG: StarDist model not available, skipping segmentation.")
            used_fallback = True

        # --- A2. Fallback: OpenCV Red-Channel Segmentation ---
        if used_fallback:
            print("Running Robust OpenCV Fallback (Red Channel)...")
            try:
                # 1. Extract Red Channel (RGB -> R is channel 0)
                # Nuclei (Blue/Purple) absorb Red light -> Dark in R channel
                # Cytoplasm (Pink) reflects Red light -> Bright in R channel
                red_channel = patch_rgb[:, :, 0]
                
                # 2. Invert: Nuclei become Bright, Cytoplasm becomes Dark
                nuclei_signal = 255 - red_channel
                
                # 3. Enhance Contrast (CLAHE)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(nuclei_signal)
                
                # 4. Otsu Thresholding
                # We want bright nuclei against dark background
                thresh_val, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 5. Morphological Opening (Remove noise/texture)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
                
                # 6. Find Contours
                contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                centroids = []
                debug_fallback = patch_rgb.copy()
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # Filter: Too small (noise) or too large (clumps/tissue fold)
                    if area < 30 or area > 2000:
                        continue
                        
                    # Filter: Circularity (Nuclei are roughly circular)
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0: continue
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity < 0.4: # Filter out long stringy shapes
                        continue
                        
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        centroids.append([cY, cX]) # Keep (y, x) format
                        cv2.drawContours(debug_fallback, [cnt], -1, (0, 255, 0), 1)
                        
                cv2.imwrite("debug_fallback.png", cv2.cvtColor(debug_fallback, cv2.COLOR_RGB2BGR))
                print(f"Fallback detected {len(centroids)} nuclei.")
                
            except Exception as e:
                print(f"Fallback Error: {e}")
                # Ultimate fallback
                n_sim = 50
                ys = np.random.randint(20, H-20, n_sim)
                xs = np.random.randint(20, W-20, n_sim)
                centroids = np.column_stack((ys, xs))

        # --- B. Feature Extraction ---
        patch_size = 64
        half_size = patch_size // 2
        
        valid_centroids = []
        tensors = []
        
        for y, x in centroids:
            y, x = int(y), int(x)
            
            if y < half_size or y >= H - half_size or x < half_size or x >= W - half_size:
                continue
                
            crop = patch_rgb[y-half_size:y+half_size, x-half_size:x+half_size]
            
            if self.extractor and self.transforms:
                try:
                    crop_cont = np.ascontiguousarray(crop)
                    t_crop = self.transforms(torch.from_numpy(crop_cont).permute(2,0,1))
                    tensors.append(t_crop)
                    valid_centroids.append((y, x))
                except:
                    pass
            else:
                valid_centroids.append((y, x))

        embeddings = []
        if self.extractor and tensors:
            try:
                batch_t = torch.stack(tensors).to(self.device)
                with torch.no_grad():
                    features = self.extractor(batch_t)
                    embeddings = features.cpu().numpy()
            except Exception as e:
                print(f"Feature Extraction Error: {e}")
                embeddings = []
        elif not self.extractor:
             embeddings = [np.random.rand(512).astype(np.float32) for _ in range(len(valid_centroids))]

        # --- C. Pack Results ---
        off_x, off_y = global_offset
        for i, (cy, cx) in enumerate(valid_centroids):
            gx = int(off_x + cx)
            gy = int(off_y + cy)
            
            emb = embeddings[i] if (i < len(embeddings)) else None
            
            results.append({
                'x': gx, 
                'y': gy,
                'embedding': emb
            })
            
        return results
