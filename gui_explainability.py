from PySide6.QtWidgets import QWidget, QVBoxLayout, QToolTip
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QMouseEvent, QPolygonF
import numpy as np
import matplotlib
matplotlib.use('QtAgg') # Ensure correct backend for PySide6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2

# Constants for visuals
COLOR_TUMOR = '#E74C3C'  # Red-ish
COLOR_NORMAL = '#3498DB' # Blue-ish

class TooltipMixin:
    """
    Mixin to add Smart Tooltip functionality to the main image viewer.
    Requires the host class to have:
    - self.get_visible_cells(): Returns list of cell dicts/objects with 'contour', 'prediction', 'confidence', 'area', etc.
    - self.widget_to_wsi_coords(qpoint): Transforms mouse pos to WSI coords.
    """

    def setup_tooltip(self):
        """Call in __init__."""
        self.setMouseTracking(True)
        
        # Debounce timer for performance
        self.hover_timer = QTimer(self)
        self.hover_timer.setSingleShot(True)
        self.hover_timer.setInterval(50) # 50ms delay
        self.hover_timer.timeout.connect(self._check_hover_target)
        
        self.last_mouse_pos = None

    def mouseMoveEvent(self, event: QMouseEvent):
        # Delegate to super first if needed, or handle here
        super().mouseMoveEvent(event)
        
        self.last_mouse_pos = event.pos()
        # Restart timer to debounce expensive lookup
        self.hover_timer.start()

    def _check_hover_target(self):
        if not self.last_mouse_pos:
            return

        # 1. Transform Coords
        # Host class must implement this mapping
        if not hasattr(self, 'widget_to_wsi_coords'):
            return
            
        wsi_pos = self.widget_to_wsi_coords(self.last_mouse_pos)
        if not wsi_pos:
            return
        
        wx, wy = wsi_pos
        
        # 2. Get Candidates (Visible Cells)
        if not hasattr(self, 'get_visible_cells'):
            return
            
        cells = self.get_visible_cells()
        if not cells:
            return

        # 3. Find Match
        # Optimization: Simple bounding box check first, then Polygon
        matched_cell = None
        
        for cell in cells:
            # Assuming cell structure has 'contour' (list of points or numpy array)
            # and potentially pre-calculated bbox
            contour = cell.get('contour')
            if contour is None: 
                continue

            # Quick bbox check
            # contour shape: (N, 2)
            min_x, min_y = np.min(contour, axis=0)
            max_x, max_y = np.max(contour, axis=0)
            
            if not (min_x <= wx <= max_x and min_y <= wy <= max_y):
                continue
                
            # Precise Polygon check using QPolygonF or cv2.pointPolygonTest
            # cv2 is faster for numpy arrays
            dist = cv2.pointPolygonTest(contour.astype(np.float32), (wx, wy), False)
            if dist >= 0: # Inside or on edge
                matched_cell = cell
                break
        
        # 4. Show Tooltip
        if matched_cell:
            self._show_smart_tooltip(matched_cell)
        else:
            QToolTip.hideText()

    def _show_smart_tooltip(self, cell):
        # Extract Data
        pred = cell.get('prediction', 0) # 0: Normal, 1: Tumor
        prob = cell.get('confidence', 0.0)
        
        # Calculate Morphology if missing
        contour = cell.get('contour')
        area = cell.get('area')
        if area is None:
            area = cv2.contourArea(contour)
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            circularity = 0
        else:
            circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Logic for Explanatory Text
        category = "Tumor" if pred == 1 else "Normal"
        color_hex = COLOR_TUMOR if pred == 1 else COLOR_NORMAL
        
        tags = []
        if area > 800: # Arbitrary threshold for example
            tags.append("(Enlarged)")
        if circularity < 0.6:
            tags.append("(Irregular)")
        
        tag_text = " ".join(tags)

        # HTML Formatting
        html = f"""
        <div style='background-color: white; border: 1px solid gray; padding: 5px;'>
            <h3 style='margin: 0; color: {color_hex};'>{category} <small>({prob:.1%})</small></h3>
            <hr style='margin: 5px 0;'>
            <table style='font-size: 11px;'>
                <tr><td><b>Area:</b></td><td>{int(area)} px² {tag_text}</td></tr>
                <tr><td><b>Circularity:</b></td><td>{circularity:.2f}</td></tr>
            </table>
        </div>
        """
        
        # Show at global screen position
        global_pos = self.mapToGlobal(self.last_mouse_pos)
        QToolTip.showText(global_pos, html, self)


class DashboardWidget(QWidget):
    """
    Real-time statistics dashboard for the sidebar.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create Matplotlib Figure
        self.figure = Figure(figsize=(3, 6), facecolor='#f0f0f0') # Slim vertical
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Subplots
        self.ax_pie = self.figure.add_subplot(211)
        self.ax_hist = self.figure.add_subplot(212)
        
        self.figure.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.3)
        
        # Init empty
        self.update_charts([])

    def update_charts(self, cells_data):
        """
        Refreshes charts with new data.
        cells_data: List of dicts/objects with 'prediction' and 'confidence'.
        """
        self.ax_pie.clear()
        self.ax_hist.clear()
        
        if not cells_data:
            self.ax_pie.text(0.5, 0.5, "No Data", ha='center')
            self.canvas.draw()
            return

        # Prepare Data
        # Ensure we handle object attributes or dict keys
        try:
            predictions = [c['prediction'] for c in cells_data]
            confidences = [c['confidence'] for c in cells_data]
        except TypeError:
            # Fallback for objects
            predictions = [c.prediction for c in cells_data]
            confidences = [c.confidence for c in cells_data]

        predictions = np.array(predictions)
        confidences = np.array(confidences)

        # --- Chart A: Pie (Tumor vs Normal) ---
        n_tumor = np.sum(predictions == 1)
        n_normal = len(predictions) - n_tumor
        
        labels = ['Normal', 'Tumor']
        sizes = [n_normal, n_tumor]
        colors = [COLOR_NORMAL, COLOR_TUMOR]
        
        self.ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                        startangle=90, wedgeprops={'edgecolor': 'white'})
        self.ax_pie.set_title("Class Distribution", fontsize=10, fontweight='bold')

        # --- Chart B: Confidence Histogram ---
        # We want to see how confident the model is.
        # Plot tumor confidence in Red, Normal in Blue (stacked or overlaid)
        
        tumor_conf = confidences[predictions == 1]
        normal_conf = confidences[predictions == 0]
        
        self.ax_hist.hist([normal_conf, tumor_conf], bins=10, range=(0.5, 1.0), 
                          color=[COLOR_NORMAL, COLOR_TUMOR], label=labels, 
                          stacked=True, alpha=0.8)
        
        self.ax_hist.set_title("Model Confidence", fontsize=10, fontweight='bold')
        self.ax_hist.set_xlabel("Probability", fontsize=8)
        self.ax_hist.set_ylabel("Count", fontsize=8)
        self.ax_hist.legend(fontsize=7)
        self.ax_hist.grid(axis='y', linestyle='--', alpha=0.5)

        self.canvas.draw()
