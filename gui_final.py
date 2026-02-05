import sys
import os
import sqlite3
import numpy as np
import cv2
import pickle
from functools import partial

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QDockWidget, 
                               QToolBar, QMessageBox, QSizePolicy,
                               QFileDialog, QMenuBar, QMenu, QButtonGroup, QTextBrowser,
                               QInputDialog)
from PySide6.QtCore import Qt, QTimer, QPoint, QRect, QSize, QMutex
from PySide6.QtGui import (
    QAction,
    QPainter,
    QColor,
    QPen,
    QBrush,
    QIcon,
    QPixmap,
    QImage,
    QKeySequence,
    QFont,
    QCursor
)

# --- Project Dependencies ---
try:
    from wsi_handler import WSILoader
except ImportError:
    print("Warning: wsi_handler.py not found. WSI features will be limited.")
    WSILoader = None

# --- Feature Modules ---
try:
    from gui_batch_annotation import BatchAnnotationMixin
except ImportError:
    print("Warning: gui_batch_annotation.py not found. Batch annotation disabled.")
    BatchAnnotationMixin = object

try:
    from gui_explainability import TooltipMixin, DashboardWidget
except ImportError:
    print("Warning: gui_explainability.py not found. Tooltips/Dashboard disabled.")
    TooltipMixin = object
    DashboardWidget = None

try:
    from gui_virtual_cytometry import VirtualCytometryWindow
except ImportError:
    print("Warning: gui_virtual_cytometry.py not found. Virtual Cytometry disabled.")
    VirtualCytometryWindow = None

try:
    from report_engine import ReportEngine
except ImportError:
    print("Warning: report_engine.py not found. Reporting disabled.")
    ReportEngine = None

try:
    from learner_extension import predict_whole_slide
except ImportError:
    predict_whole_slide = None

try:
    from gui_analysis_integration import AnalysisWorker
except ImportError:
    AnalysisWorker = None

# --- Constants ---
MODE_PAN = 'PAN'
MODE_BOX = 'BOX_SELECT'
MODE_CLICK = 'SINGLE_CLICK'

class FinalPathologyGUI(QMainWindow, BatchAnnotationMixin, TooltipMixin):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Gemini Pathology AI - Human-in-the-Loop System")
        self.resize(1600, 1000)
        
        # --- Thread Safety ---
        self.data_lock = QMutex()
        
        # --- State ---
        self.wsi_path = None
        self.db_path = None
        self.wsi_handler = None
        
        self.zoom_level = 0 
        self.view_center = (0, 0)
        self.view_size = (1024, 768)
        self.highlighted_ids = set()
        
        # Interaction State
        self.interaction_mode = MODE_PAN
        self.last_mouse_pos = None
        self.is_panning = False
        
        # Analysis State (for Guide)
        self.has_analyzed = False
        self.has_labeled = False
        
        # --- UI Setup ---
        self._init_ui()
        self._setup_menubar()
        
        # --- Mixin Setup ---
        self.setup_batch_annotation() 
        self.setup_tooltip()
        
        # --- Cytometry Window State ---
        self.cytometry_window = None

        # --- Auto Open on Startup ---
        QTimer.singleShot(100, self.open_file_dialog)

    def _init_ui(self):
        # --- Toolbar (Top, Prominent) ---
        self.toolbar = QToolBar("Main Workflow")
        self.toolbar.setIconSize(QSize(32, 32))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon) # Icon + Text
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)
        
        # 1. Pan/Navigate
        self.act_pan = QAction("✋ 浏览模式 (Pan)", self)
        self.act_pan.setCheckable(True)
        self.act_pan.setChecked(True)
        self.act_pan.triggered.connect(lambda: self.set_mode(MODE_PAN))
        self.toolbar.addAction(self.act_pan)
        
        self.toolbar.addSeparator()

        # 2. Analyze (Red/Bold)
        self.act_analyze = QAction("⚡ 分析当前视野 (Analyze)", self)
        self.act_analyze.triggered.connect(self.run_analysis)
        self.toolbar.addAction(self.act_analyze)
        
        # 3. Box Select
        self.act_box = QAction("📏 框选标注 (Box Select)", self)
        self.act_box.setCheckable(True)
        self.act_box.triggered.connect(lambda: self.set_mode(MODE_BOX))
        self.toolbar.addAction(self.act_box)

        # 4. Single Click (New)
        self.act_click = QAction("👆 单点标注 (Click)", self)
        self.act_click.setCheckable(True)
        self.act_click.triggered.connect(lambda: self.set_mode(MODE_CLICK))
        self.toolbar.addAction(self.act_click)
        
        self.toolbar.addSeparator()

        # 5. Cytometry
        self.act_cytometry = QAction("🔬 打开流式 (Cytometry)", self)
        self.act_cytometry.triggered.connect(self.open_cytometry_view)
        self.toolbar.addAction(self.act_cytometry)

        # 6. Report
        self.act_report = QAction("📄 生成报告 (Report)", self)
        self.act_report.triggered.connect(self.generate_final_report)
        self.toolbar.addAction(self.act_report)
        
        # --- Central Canvas ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.canvas_layout = QVBoxLayout(self.central_widget)
        self.canvas_layout.setContentsMargins(0,0,0,0)
        
        self.image_label = QLabel("Initializing...\nPlease Open a WSI File.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #202020; color: #AAA; font-size: 18px; font-weight: bold;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMouseTracking(True) 
        self.canvas_layout.addWidget(self.image_label)

        # --- Sidebar (Right) ---
        self.sidebar = QDockWidget("Dashboard & Guide", self)
        self.sidebar.setFeatures(QDockWidget.NoDockWidgetFeatures) # Fixed
        self.sidebar.setAllowedAreas(Qt.RightDockWidgetArea)
        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)
        self.sidebar_layout.setSpacing(15)
        
        # A. Onboarding Guide (Top of Sidebar)
        self.guide_box = QTextBrowser()
        self.guide_box.setMaximumHeight(150)
        self.guide_box.setStyleSheet("""
            background-color: #E8F8F5; 
            border: 2px solid #1ABC9C; 
            border-radius: 5px; 
            color: #2C3E50;
            padding: 5px;
        """)
        self.update_guide()
        self.sidebar_layout.addWidget(QLabel("<b>💡 操作指引 (Guide):</b>"))
        self.sidebar_layout.addWidget(self.guide_box)
        
        # B. Dashboard
        if DashboardWidget:
            self.dashboard = DashboardWidget()
            self.sidebar_layout.addWidget(self.dashboard)
        else:
            self.sidebar_layout.addWidget(QLabel("Dashboard Missing"))

        # C. Model Controls
        self.sidebar_layout.addStretch()
        self.lbl_model_status = QLabel("Model: ResNet18 (Active Learning Ready)")
        self.lbl_model_status.setStyleSheet("color: #666; font-size: 11px;")
        self.sidebar_layout.addWidget(self.lbl_model_status)

        self.sidebar.setWidget(self.sidebar_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.sidebar)

    def _setup_menubar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        
        open_action = QAction("Open WSI...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        
        file_menu.addAction("Exit", self.close)

    # --- Mode & Interaction Management ---

    def set_mode(self, mode):
        self.interaction_mode = mode
        
        # Reset all checks
        self.act_pan.setChecked(mode == MODE_PAN)
        self.act_box.setChecked(mode == MODE_BOX)
        self.act_click.setChecked(mode == MODE_CLICK)
        
        if mode == MODE_PAN:
            self.setCursor(Qt.OpenHandCursor)
            self.toggle_box_select(False) 
        elif mode == MODE_BOX:
            self.setCursor(Qt.CrossCursor)
            self.toggle_box_select(True)
        elif mode == MODE_CLICK:
            self.setCursor(Qt.PointingHandCursor)
            self.toggle_box_select(False)

    def update_guide(self):
        """Updates the text browser based on state."""
        style = "<style>h3{margin:0; color:#16A085;} p{margin:5px 0; font-size:13px;}</style>"
        
        if not self.wsi_path:
            html = f"{style}<h3>👋 Welcome!</h3><p>请点击左上角 <b>File -> Open WSI</b> 打开切片文件。</p>"
        elif not self.has_analyzed:
            html = f"{style}<h3>Step 1: 发现细胞</h3><p>1. 滚动鼠标滚轮放大图片。</p><p>2. 移动到感兴趣区域。</p><p>3. 点击顶部工具栏的 <b>⚡ 分析当前视野</b> 按钮。</p>"
        elif not self.has_labeled:
            html = f"{style}<h3>Step 2: 教会 AI</h3><p>1. 切换到 <b>📏 框选</b> 或 <b>👆 单点</b> 模式。</p><p>2. 标记几个细胞为 Tumor/Normal。</p><p>3. 观察 AI 如何举一反三！</p>"
        else:
            html = f"{style}<h3>Step 3: 协同进化</h3><p>AI 已根据您的反馈更新预测（透明色）。</p><p>- 继续标注修正 AI 的错误。</p><p>- 或点击 <b>📄 生成报告</b>。</p>"
            
        self.guide_box.setHtml(html)

    # --- Mouse Events (Pan, Box, Click) ---

    def mousePressEvent(self, event):
        if self.interaction_mode == MODE_PAN and event.button() == Qt.LeftButton:
            self.is_panning = True
            self.last_mouse_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
        elif self.interaction_mode == MODE_BOX:
            self.on_mouse_press(event)
        elif self.interaction_mode == MODE_CLICK and event.button() == Qt.LeftButton:
            # Handle Single Click Labeling
            self.handle_single_click_labeling(event.position().toPoint())
        
        super().mousePressEvent(event)

    def handle_single_click_labeling(self, pos):
        if not self.db_path: return
        
        # Map to WSI coords
        wx, wy = self.widget_to_wsi_coords(pos)
        
        # Find nearest cell within tolerance (e.g. 20px)
        tol = 20
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple bounding box query first
        cursor.execute("SELECT rowid, x, y FROM cells WHERE x BETWEEN ? AND ? AND y BETWEEN ? AND ?", 
                       (wx-tol, wx+tol, wy-tol, wy+tol))
        rows = cursor.fetchall()
        
        if not rows:
            conn.close()
            return
            
        # Find closest
        best_id = None
        min_dist = float('inf')
        
        for rid, cx, cy in rows:
            dist = (cx - wx)**2 + (cy - wy)**2
            if dist < min_dist:
                min_dist = dist
                best_id = rid
        
        if best_id:
            # Ask User
            items = ["Tumor", "Normal"]
            item, ok = QInputDialog.getItem(self, "Label Cell", "Classify this cell:", items, 0, False)
            
            if ok and item:
                lbl_val = 1 if item == "Tumor" else 0
                print(f"Labeling cell {best_id} as {lbl_val}")
                
                # Update DB
                cursor.execute("UPDATE cells SET label = ?, prediction = ? WHERE rowid = ?", (lbl_val, lbl_val, best_id))
                conn.commit()
                conn.close()
                
                # Trigger Active Learning
                self.trigger_active_learning()
                self.has_labeled = True
                self.update_guide()
        else:
            conn.close()

    def trigger_active_learning(self):
        try:
            from active_learning import train_on_labeled_cells, predict_unlabeled_cells
            learner = train_on_labeled_cells(self.db_path)
            predict_unlabeled_cells(self.db_path, learner)
            self.update_view()
        except Exception as e:
            print(f"Active Learning Error: {e}")

    def mouseMoveEvent(self, event):
        if self.interaction_mode == MODE_PAN and self.is_panning:
            delta = event.position().toPoint() - self.last_mouse_pos
            self.last_mouse_pos = event.position().toPoint()
            
            if self.wsi_handler:
                scale = self.wsi_handler.slide.level_downsamples[self.zoom_level]
                dx_wsi = delta.x() * scale
                dy_wsi = delta.y() * scale
                self.view_center = (
                    int(self.view_center[0] - dx_wsi),
                    int(self.view_center[1] - dy_wsi)
                )
                self.update_view()
        elif self.interaction_mode == MODE_BOX:
            self.on_mouse_move(event)
            
        super().mouseMoveEvent(event) 

    def mouseReleaseEvent(self, event):
        if self.interaction_mode == MODE_PAN and event.button() == Qt.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.OpenHandCursor)
        elif self.interaction_mode == MODE_BOX:
            self.on_mouse_release(event)
            self.has_labeled = True 
            self.update_guide()
            
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if not self.wsi_handler: return
        angle = event.angleDelta().y()
        max_level = self.wsi_handler.slide.level_count - 1
        
        if angle > 0: self.zoom_level = max(0, self.zoom_level - 1)
        else: self.zoom_level = min(max_level, self.zoom_level + 1)
            
        self.update_view()

    # --- Analysis & Core Logic ---

    def run_analysis(self):
        if not self.wsi_handler or not self.db_path:
            QMessageBox.warning(self, "Error", "No slide loaded.")
            return
        
        scale = self.wsi_handler.slide.level_downsamples[self.zoom_level]
        wsi_w = int(self.view_size[0] * scale)
        wsi_h = int(self.view_size[1] * scale)
        
        if wsi_w > 5000 or wsi_h > 5000:
            QMessageBox.warning(self, "Area Too Large", "Please zoom in further to analyze.")
            return

        tl_x = max(0, int(self.view_center[0] - wsi_w / 2))
        tl_y = max(0, int(self.view_center[1] - wsi_h / 2))
        
        self.setCursor(Qt.WaitCursor)
        self.act_analyze.setEnabled(False)
        self.image_label.setText("Analyzing... Please Wait") 
        
        if AnalysisWorker:
            self.analysis_worker = AnalysisWorker(self.wsi_handler, (tl_x, tl_y, wsi_w, wsi_h), self.db_path)
            self.analysis_worker.finished.connect(self.on_analysis_finished)
            self.analysis_worker.error.connect(self.on_analysis_error)
            self.analysis_worker.start()
        else:
            self.setCursor(Qt.ArrowCursor)
            self.act_analyze.setEnabled(True)
            QMessageBox.critical(self, "Error", "Analysis module missing.")

    def on_analysis_finished(self, count):
        self.setCursor(Qt.ArrowCursor if self.interaction_mode==MODE_PAN else Qt.CrossCursor)
        self.act_analyze.setEnabled(True)
        self.image_label.setText("") 
        
        self.has_analyzed = True
        self.update_guide()
        
        QMessageBox.information(self, "Analysis Complete", f"Successfully detected {count} cells.")
        self.update_view()

    def on_analysis_error(self, msg):
        self.setCursor(Qt.ArrowCursor)
        self.act_analyze.setEnabled(True)
        self.image_label.setText("")
        QMessageBox.critical(self, "Analysis Failed", msg)

    # --- Standard Methods (Load, Update, Coords) ---
    
    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Slide", "", "WSI Files (*.svs *.ndpi *.tiff *.tif);;All (*)")
        if file_path:
            self.load_wsi(file_path)

    def load_wsi(self, path):
        if not WSILoader: return
        try:
            self.wsi_handler = WSILoader(path)
            dim = self.wsi_handler.slide.dimensions
            self.view_center = (dim[0] // 2, dim[1] // 2)
            self.zoom_level = self.wsi_handler.slide.level_count - 2
            self.zoom_level = max(0, self.zoom_level)
            
            if hasattr(self.wsi_handler, 'detect_tissue_regions'):
                pts, _ = self.wsi_handler.detect_tissue_regions()
                if pts: self.view_center = pts[0]
            
            self.wsi_path = path
            self.db_path = f"{os.path.splitext(os.path.basename(path))[0]}_data.db"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cells'")
            if not cursor.fetchone():
                cursor.execute("CREATE TABLE cells (x INTEGER, y INTEGER, label INTEGER DEFAULT -1, prediction INTEGER DEFAULT -1, tumor_prob REAL DEFAULT 0.0, feature_data BLOB)")
                conn.commit()
            conn.close()
            
            self.update_guide()
            self.update_view()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def update_view(self):
        if not self.wsi_handler: return
        
        if not self.data_lock.tryLock(): return 

        try:
            w, h = self.central_widget.width(), self.central_widget.height()
            self.view_size = (w, h)
            self.zoom_level = max(0, min(self.zoom_level, self.wsi_handler.slide.level_count - 1))
            
            img_np = self.wsi_handler.read_region_as_tensor(self.view_center[0], self.view_center[1], self.zoom_level, (w, h))
            if img_np is None: return
            
            h_img, w_img, _ = img_np.shape
            q_img = QImage(img_np.data, w_img, h_img, 3*w_img, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            visible_cells = self.get_visible_cells()
            if self.dashboard and visible_cells: self.dashboard.update_charts(visible_cells)
            
            for c in visible_cells:
                cx, cy = self.wsi_to_widget_coords(c['x'], c['y'])
                
                # Default: Unlabeled/White
                pen_color = QColor(255, 255, 255) 
                brush_color = QColor(255, 255, 255, 30)
                radius = 4
                pen_width = 1

                label = c['label']
                pred = c['prediction']

                if c['id'] in self.highlighted_ids:
                    pen_color = QColor(255, 255, 0) # Yellow Highligh
                    brush_color = QColor(255, 255, 0, 100)
                    radius = 8
                    pen_width = 3
                
                # Human Labeled (Override Prediction)
                elif label != -1:
                    pen_width = 3
                    if label == 1: # Tumor (Human)
                        pen_color = QColor(255, 0, 0)
                        brush_color = QColor(255, 0, 0, 200) # Solid-ish
                    else: # Normal (Human)
                        pen_color = QColor(0, 255, 255)
                        brush_color = QColor(0, 255, 255, 200) # Solid-ish
                
                # AI Prediction
                elif pred != -1:
                    pen_width = 2
                    if pred == 1: # Tumor (AI)
                        pen_color = QColor(255, 100, 100) # Lighter Red
                        brush_color = QColor(255, 0, 0, 80) # Transparent
                    else: # Normal (AI)
                        pen_color = QColor(100, 255, 255) # Lighter Cyan
                        brush_color = QColor(0, 255, 255, 80) # Transparent

                painter.setPen(QPen(pen_color, pen_width))
                painter.setBrush(QBrush(brush_color))
                painter.drawEllipse(QPoint(cx, cy), radius, radius)
                
            painter.end()
            self.image_label.setPixmap(pixmap)
        finally:
            self.data_lock.unlock()

    def get_visible_cells(self):
        if not self.db_path or not os.path.exists(self.db_path): return []
        scale = self.wsi_handler.slide.level_downsamples[self.zoom_level]
        wsi_w = self.view_size[0] * scale
        wsi_h = self.view_size[1] * scale
        
        x1, y1 = self.view_center[0] - wsi_w/2, self.view_center[1] - wsi_h/2
        x2, y2 = x1 + wsi_w, y1 + wsi_h
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT rowid as id, x, y, label, prediction, tumor_prob, feature_data FROM cells WHERE x >= ? AND x <= ? AND y >= ? AND y <= ?", (x1, x2, y1, y2))
            rows = cursor.fetchall()
        except: return []
        finally: conn.close()
        
        cells = []
        for r in rows:
            c = dict(r)
            c['contour'] = np.array([[c['x']-20, c['y']-20], [c['x']+20, c['y']-20], [c['x']+20, c['y']+20], [c['x']-20, c['y']+20]])
            c['area'] = 400
            c['confidence'] = c['tumor_prob'] or 0.5
            try:
                c['embedding'] = pickle.loads(c['feature_data']) if c['feature_data'] else None
            except:
                c['embedding'] = None
            cells.append(c)
        return cells

    def widget_to_wsi_coords(self, qpoint):
        if not self.wsi_handler: return None
        wx, wy = qpoint.x(), qpoint.y()
        scale = self.wsi_handler.slide.level_downsamples[self.zoom_level]
        dx = (wx - self.view_size[0]/2) * scale
        dy = (wy - self.view_size[1]/2) * scale
        return int(self.view_center[0] + dx), int(self.view_center[1] + dy)

    def wsi_to_widget_coords(self, wx, wy):
        if not self.wsi_handler: return 0,0
        scale = self.wsi_handler.slide.level_downsamples[self.zoom_level]
        dx = (wx - self.view_center[0]) / scale
        dy = (wy - self.view_center[1]) / scale
        return int(self.view_size[0]/2 + dx), int(self.view_size[1]/2 + dy)
        
    def toggle_box_select_wrapper(self, checked): pass 
    def train_model(self): self.trigger_active_learning()
    def train_step(self): self.trigger_active_learning()
    def update_overlay(self): self.update_view()
    
    def open_cytometry_view(self):
        if not VirtualCytometryWindow: return
        if not self.cytometry_window:
            self.cytometry_window = VirtualCytometryWindow()
            self.cytometry_window.selection_changed.connect(self.on_cytometry_selection)
            self.cytometry_window.batch_label_requested.connect(self.on_cytometry_batch_label)
        
        cells = self.get_visible_cells()
        if not cells: 
            QMessageBox.info(self, "Info", "No cells visible.")
            return
        
        emb = [c['embedding'].flatten() for c in cells if c['embedding'] is not None]
        ids = [c['id'] for c in cells if c['embedding'] is not None]
        lbl = [c['prediction'] for c in cells if c['embedding'] is not None]
        
        if emb:
            self.cytometry_window.set_data(np.array(emb), np.array(lbl), ids)
            self.cytometry_window.show()

    def on_cytometry_selection(self, cids):
        self.highlighted_ids = set(cids)
        self.update_view()
        
    def on_cytometry_batch_label(self, cids, lbl):
        if not self.db_path: return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"UPDATE cells SET label=?, prediction=? WHERE rowid IN ({','.join(['?']*len(cids))})", (lbl, lbl, *cids))
        conn.commit()
        conn.close()
        self.trigger_active_learning()
        self.update_view()

    def generate_final_report(self):
        if not ReportEngine: return
        engine = ReportEngine(self.db_path, self.wsi_handler)
        dim = self.wsi_handler.slide.dimensions
        path = engine.generate_heatmap(dim[0], dim[1])
        engine.export_final_report(path)
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(engine.output_dir)}/final_diagnosis.html")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FinalPathologyGUI()
    window.show()
    sys.exit(app.exec())