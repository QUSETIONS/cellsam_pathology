import sys
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QMenu, QDialog, QMessageBox)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QAction, QCursor

# Try importing dependencies
try:
    import pyqtgraph as pg
    from sklearn.decomposition import PCA
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    MISSING_DEP_MSG = str(e)

class VirtualCytometryWindow(QWidget):
    """
    A floating window showing a PCA scatter plot of cell embeddings.
    Allows interactive selection and batch labeling.
    """
    
    # Signals
    selection_changed = Signal(list) 
    batch_label_requested = Signal(list, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Virtual Flow Cytometry (Feature Space)")
        self.resize(600, 500)
        
        if not HAS_DEPS:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel(f"Missing dependencies:\n{MISSING_DEP_MSG}\n\nPlease install: pip install pyqtgraph scikit-learn"))
            return

        self.cell_ids = []
        self.pca_coords = None
        self.labels = None
        self.selected_indices = set()

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Toolbar
        toolbar = QHBoxLayout()
        self.lbl_status = QLabel("Ready. Load data to view.")
        toolbar.addWidget(self.lbl_status)
        toolbar.addStretch() 
        
        self.btn_select_mode = QPushButton("Select Mode: Pan/Zoom")
        self.btn_select_mode.setCheckable(True)
        self.btn_select_mode.clicked.connect(self._toggle_select_mode)
        toolbar.addWidget(self.btn_select_mode)
        layout.addLayout(toolbar)

        # 2. Plot Widget (PyQtGraph)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'PC 2')
        self.plot_widget.setLabel('bottom', 'PC 1')
        self.plot_widget.showGrid(x=True, y=True)
        
        self.scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None))
        self.scatter.sigClicked.connect(self._on_points_clicked)
        self.plot_widget.addItem(self.scatter)
        
        layout.addWidget(self.plot_widget)
        
        self.selection_box = pg.PlotCurveItem(pen=pg.mkPen('k', style=Qt.DashLine))
        self.plot_widget.addItem(self.selection_box)
        
        # Hook mouse events
        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self._on_scene_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_click)
        
        self.is_dragging = False
        self.drag_start = None
        self.drag_end = None

        # Install event filter for dragging logic
        self.plot_widget.viewport().installEventFilter(self)

    def set_data(self, embeddings, labels, cell_ids):
        if not HAS_DEPS: return
        num_cells = len(cell_ids)
        self.lbl_status.setText(f"Processing {num_cells} cells...")
        self.cell_ids = np.array(cell_ids)
        self.labels = np.array(labels)
        
        if num_cells < 2: return

        try:
            pca = PCA(n_components=2)
            self.pca_coords = pca.fit_transform(embeddings)
        except Exception as e:
            self.lbl_status.setText(f"PCA Error: {e}")
            return

        self._refresh_plot()
        self.lbl_status.setText(f"Showing {num_cells} cells.")

    def _refresh_plot(self):
        if self.pca_coords is None: return

        brushes = []
        for i, lbl in enumerate(self.labels):
            if i in self.selected_indices:
                brushes.append(pg.mkBrush('y'))
            elif lbl == 1:
                brushes.append(pg.mkBrush(231, 76, 60, 150))
            elif lbl == 0:
                brushes.append(pg.mkBrush(52, 152, 219, 150))
            else:
                brushes.append(pg.mkBrush(150, 150, 150, 150))

        self.scatter.setData(
            x=self.pca_coords[:, 0],
            y=self.pca_coords[:, 1],
            brush=brushes,
            hoverable=True,
            tip=None
        )

    def _toggle_select_mode(self, checked):
        if checked:
            self.btn_select_mode.setText("Select Mode: Box Selection")
            self.plot_widget.setMouseEnabled(x=False, y=False)
            self.plot_widget.setCursor(Qt.CrossCursor)
        else:
            self.btn_select_mode.setText("Select Mode: Pan/Zoom")
            self.plot_widget.setMouseEnabled(x=True, y=True)
            self.plot_widget.setCursor(Qt.ArrowCursor)
            self.selection_box.setData([], [])

    # --- Mouse Interaction ---

    def _on_scene_mouse_moved(self, evt):
        # Placeholder to prevent crash
        pass

    def _on_mouse_click(self, evt):
        # Placeholder to prevent crash
        pass

    def eventFilter(self, source, event):
        if source == self.plot_widget.viewport() and self.btn_select_mode.isChecked():
            if event.type() == event.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.is_dragging = True
                    self.drag_start = self.plot_widget.plotItem.vb.mapSceneToView(event.pos())
                    return True
            elif event.type() == event.MouseMove:
                if self.is_dragging:
                    self.drag_end = self.plot_widget.plotItem.vb.mapSceneToView(event.pos())
                    self._update_selection_box()
                    return True
            elif event.type() == event.MouseButtonRelease:
                if self.is_dragging and event.button() == Qt.LeftButton:
                    self.is_dragging = False
                    self.drag_end = self.plot_widget.plotItem.vb.mapSceneToView(event.pos())
                    self._finalize_selection()
                    self.selection_box.setData([], [])
                    return True
        return super().eventFilter(source, event)

    def _update_selection_box(self):
        if self.drag_start is None or self.drag_end is None: return
        p1 = self.drag_start
        p2 = self.drag_end
        x = [p1.x(), p2.x(), p2.x(), p1.x(), p1.x()]
        y = [p1.y(), p1.y(), p2.y(), p2.y(), p1.y()]
        self.selection_box.setData(x, y)

    def _finalize_selection(self):
        if self.pca_coords is None: return
        
        x_min = min(self.drag_start.x(), self.drag_end.x())
        x_max = max(self.drag_start.x(), self.drag_end.x())
        y_min = min(self.drag_start.y(), self.drag_end.y())
        y_max = max(self.drag_start.y(), self.drag_end.y())
        
        xs = self.pca_coords[:, 0]
        ys = self.pca_coords[:, 1]
        
        mask = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
        indices = np.where(mask)[0]
        
        self.selected_indices = set(indices)
        self._refresh_plot()
        
        selected_ids = self.cell_ids[indices].tolist()
        self.selection_changed.emit(selected_ids)
        self.lbl_status.setText(f"Selected {len(selected_ids)} cells.")

    def _on_points_clicked(self, plot, points):
        if self.btn_select_mode.isChecked(): return
        pass

    def contextMenuEvent(self, event):
        if not self.selected_indices: return
        menu = QMenu(self)
        action_tumor = QAction("Mark Selection as Tumor", self)
        action_tumor.triggered.connect(lambda: self._emit_batch_label(1))
        action_normal = QAction("Mark Selection as Normal", self)
        action_normal.triggered.connect(lambda: self._emit_batch_label(0))
        menu.addAction(action_tumor)
        menu.addAction(action_normal)
        menu.exec_(event.globalPos())

    def _emit_batch_label(self, label):
        ids = self.cell_ids[list(self.selected_indices)].tolist()
        self.batch_label_requested.emit(ids, label)
        for idx in self.selected_indices:
            self.labels[idx] = label
        self._refresh_plot()
    
    def highlight_cells(self, cell_ids):
        if self.pca_coords is None: return
        id_to_idx = {cid: i for i, cid in enumerate(self.cell_ids)}
        new_selection = []
        for cid in cell_ids:
            if cid in id_to_idx:
                new_selection.append(id_to_idx[cid])
        self.selected_indices = set(new_selection)
        self._refresh_plot()