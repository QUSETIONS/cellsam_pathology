from PySide6.QtCore import QThread, Signal
from preprocessing_pipeline import NucleiPipeline, FeaturesDatabase
import traceback
import numpy as np

class AnalysisWorker(QThread):
    finished = Signal(int) # Emits number of cells found
    error = Signal(str)

    def __init__(self, wsi_handler, viewport_rect, db_path):
        """
        Args:
            wsi_handler: Instance of WSILoader (has .slide attribute)
            viewport_rect: tuple (x, y, w, h) - Top-Left Global Coords & Size (Level 0)
            db_path: Path to SQLite DB
        """
        super().__init__()
        self.wsi_handler = wsi_handler
        self.viewport_rect = viewport_rect
        self.db_path = db_path

    def run(self):
        try:
            x, y, w, h = self.viewport_rect
            
            # 1. Force Level 0 Read for Maximum Precision
            # openslide.read_region(location, level, size)
            # location is (x, y) at level 0
            # size is (w, h) of output image
            
            # Note: We assume self.wsi_handler.slide is the OpenSlide object
            region = self.wsi_handler.slide.read_region((x, y), 0, (w, h))
            
            # Convert to RGB Numpy Array (OpenSlide returns RGBA PIL Image)
            patch_image = np.array(region.convert("RGB"))
            
            # 2. Initialize Pipeline (Singleton)
            pipeline = NucleiPipeline()
            
            # 3. Process
            # patch_image is now guaranteed Level 0 High-Res
            # offset is (x, y) for global coordinate mapping
            cells_data = pipeline.process_patch(patch_image, global_offset=(x, y))
            
            # 4. Save to DB
            if cells_data:
                db = FeaturesDatabase(self.db_path)
                db.insert_batch(cells_data)
                
            self.finished.emit(len(cells_data))
            
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
