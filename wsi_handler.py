import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Configure OpenSlide DLL path for Windows
# This must happen BEFORE importing openslide
def _configure_openslide():
    if os.name == 'nt':
        # Assuming the structure created by setup_openslide.py
        # D:\necio\cellsam\openslide_bin\openslide-win64-20171122\bin
        base_dir = Path(__file__).parent.absolute()
        openslide_bin_path = base_dir / "openslide_bin" / "openslide-win64-20171122" / "bin"
        
        if openslide_bin_path.exists():
            try:
                os.add_dll_directory(str(openslide_bin_path))
            except AttributeError:
                os.environ['PATH'] = str(openslide_bin_path) + ";" + os.environ['PATH']
        else:
            print(f"Warning: OpenSlide bin directory not found at {openslide_bin_path}")

_configure_openslide()

try:
    import openslide
except ImportError as e:
    print("CRITICAL ERROR: Could not import openslide.")
    raise e

class WSILoader:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            raise FileNotFoundError(f"SVS file not found at: {path}")
        
        try:
            self.slide = openslide.OpenSlide(path)
        except Exception as e:
            print(f"Error opening slide: {e}")
            raise

    def get_info(self):
        print(f"--- WSI Info for: {os.path.basename(self.path)} ---")
        print(f"Dimensions (Level 0): {self.slide.dimensions}")
        print(f"Level count: {self.slide.level_count}")
        print(f"Level dimensions: {self.slide.level_dimensions}")
        mpp_x = self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        print(f"Microns per pixel: {mpp_x if mpp_x else 'Unknown'}")
        print("------------------------------------------------")

    def read_region_as_tensor(self, center_x, center_y, level, size):
        """
        Reads a region from the slide.
        args:
            center_x, center_y: Coordinates in Level 0 reference frame.
            level: Pyramid level to read from.
            size: (width, height) tuple of the OUTPUT image size.
        returns:
            numpy array of shape (H, W, 3) (RGB)
        """
        ds = self.slide.level_downsamples[level]
        offset_x = int(size[0] / 2 * ds)
        offset_y = int(size[1] / 2 * ds)
        
        top_left_x = int(center_x - offset_x)
        top_left_y = int(center_y - offset_y)
        
        top_left_x = max(0, top_left_x)
        top_left_y = max(0, top_left_y)

        try:
            img = self.slide.read_region((top_left_x, top_left_y), level, size)
            img = img.convert("RGB")
            img_np = np.array(img)
            return img_np
        except Exception as e:
            print(f"Error reading region: {e}")
            return None

    def detect_tissue_regions(self, thumb_size=(1024, 1024), threshold=215):
        """
        Detects tissue regions using the thumbnail.
        Returns:
            list of (center_x, center_y) in Level 0 coordinates for potential patches.
            thumb_img: The thumbnail PIL image.
        """
        # Get thumbnail
        thumb = self.slide.get_thumbnail(thumb_size)
        arr = np.array(thumb)
        gray = np.mean(arr, axis=2)
        binary = gray < threshold # Tissue is dark
        
        from scipy.ndimage import label, find_objects
        labeled, n_components = label(binary)
        
        if n_components == 0:
            return [], thumb
            
        regions = []
        slices = find_objects(labeled)
        
        # Scaling factors
        w_thumb, h_thumb = thumb.size
        w_l0, h_l0 = self.slide.dimensions
        scale_x = w_l0 / w_thumb
        scale_y = h_l0 / h_thumb
        
        # Sort by area (descending)
        regions_props = []
        for i, sl in enumerate(slices):
            if sl is None: continue
            area = np.sum(labeled[sl] == (i + 1))
            if area > 100: # Filter tiny specks
                regions_props.append((area, sl, i+1))
        
        regions_props.sort(key=lambda x: x[0], reverse=True)
        
        # Extract centroids of top 5 regions
        points = []
        for _, sl, label_idx in regions_props[:5]:
            coords = np.argwhere(labeled[sl] == label_idx)
            # coords are relative to the slice, add slice start
            # But argwhere returns (row, col) -> (y, x)
            y_local, x_local = np.mean(coords, axis=0)
            y_thumb = sl[0].start + y_local
            x_thumb = sl[1].start + x_local
            
            cx_l0 = int(x_thumb * scale_x)
            cy_l0 = int(y_thumb * scale_y)
            points.append((cx_l0, cy_l0))
            
        return points, thumb