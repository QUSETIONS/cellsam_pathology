from PySide6.QtWidgets import QWidget, QRubberBand, QMessageBox, QDialog, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QRadioButton, QButtonGroup
from PySide6.QtCore import QRect, QPoint, QSize, Qt
from PySide6.QtGui import QMouseEvent
import sqlite3
import numpy as np

class BatchAnnotationMixin:
    """
    Mixin class to add Batch Annotation (Box Selection) functionality to the Main GUI.
    Assumes the host class has:
    - self.image_label: The widget displaying the image (e.g., QLabel).
    - self.wsi_handler: Handler to get image dimensions/scaling.
    - self.db_path: Path to the SQLite database.
    - self.learner: The active learning model instance.
    - self.update_overlay(): Method to refresh the visualization.
    """

    def setup_batch_annotation(self):
        """Call this in __init__ to initialize rubber band and flags."""
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.image_label)
        self.origin = QPoint()
        self.is_selecting = False
        self.box_select_mode = False # Toggle state

        # Add toggle button to your toolbar programmatically or via Designer
        # self.btn_box_select = QAction("Box Select", self)
        # self.btn_box_select.setCheckable(True)
        # self.btn_box_select.toggled.connect(self.toggle_box_select)

    def toggle_box_select(self, checked):
        self.box_select_mode = checked
        cursor = Qt.CrossCursor if checked else Qt.ArrowCursor
        self.image_label.setCursor(cursor)

    # --- Mouse Events (Override or connect these in your main class) ---
    
    def on_mouse_press(self, event: QMouseEvent):
        if not self.box_select_mode:
            # Fallback to original click logic (single cell selection)
            # return super().mousePressEvent(event) 
            return

        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()
            self.is_selecting = True

    def on_mouse_move(self, event: QMouseEvent):
        if not self.box_select_mode or not self.is_selecting:
            return

        self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def on_mouse_release(self, event: QMouseEvent):
        if not self.box_select_mode or not self.is_selecting:
            return

        if event.button() == Qt.LeftButton:
            self.is_selecting = False
            rect = self.rubber_band.geometry()
            self.rubber_band.hide()
            
            # Minimum size check to avoid accidental clicks
            if rect.width() > 5 and rect.height() > 5:
                self.process_selection(rect)

    # --- Logic ---

    def process_selection(self, rect_widget):
        """
        1. Map widget coords to WSI coords.
        2. Query DB for cells in this region.
        3. Prompt user for label.
        4. Update DB and Retrain.
        """
        
        # 1. Coordinate Mapping (Widget -> WSI Level 0)
        # Fix: Convert corners separately
        if not hasattr(self, 'widget_to_wsi_coords'):
            print("Error: Host class missing 'widget_to_wsi_coords'")
            return

        # QRect to QPoint corners
        tl = rect_widget.topLeft()
        br = rect_widget.bottomRight()
        
        wsi_tl = self.widget_to_wsi_coords(tl)
        wsi_br = self.widget_to_wsi_coords(br)
        
        if not wsi_tl or not wsi_br:
            return

        x1, y1 = wsi_tl
        x2, y2 = wsi_br
        
        # Ensure correct order (handling negative drag)
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # 2. Query DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find cells within the box
        query = f"SELECT rowid, x, y FROM cells WHERE x >= ? AND x <= ? AND y >= ? AND y <= ?"
        cursor.execute(query, (x_min, x_max, y_min, y_max))
        cells = cursor.fetchall()
        
        if not cells:
            conn.close()
            return # No cells selected

        ids = [c[0] for c in cells]
        count = len(ids)

        # 3. Dialog
        label, ok = self.show_batch_label_dialog(count)
        
        if ok and label is not None:
            # 4. Update and Teach
            print(f"Batch tagging {count} cells as {label}...")
            
            # Update DB - Update 'label' AND reset 'prediction'
            cursor.execute(f"UPDATE cells SET label = ?, prediction = ? WHERE rowid IN ({','.join(['?']*len(ids))})", 
                           (label, label, *ids))
            conn.commit()
            conn.close() # Close before AL steps
            
            # --- Active Learning Trigger ---
            try:
                from active_learning import train_on_labeled_cells, predict_unlabeled_cells
                
                # A. Train Classifier
                learner = train_on_labeled_cells(self.db_path)
                
                # B. Propagate Predictions
                propagated_count = predict_unlabeled_cells(self.db_path, learner)
                
                msg = f"Labeled {count} cells.\nModel updated & propagated to {propagated_count} cells."
            except Exception as e:
                msg = f"Labeled {count} cells.\nActive Learning Error: {e}"
                print(msg)
            
            # Feedback
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Active Learning", msg)
            
            # Trigger generic update
            if hasattr(self, 'train_step'):
                self.train_step() 
            
            self.update_overlay() # Refresh UI
        else:
            conn.close()

    def widget_to_wsi_coords(self, rect):
        """
        Convert QRect in widget coordinates to tuple (x1, y1, x2, y2) in WSI coordinates.
        This depends heavily on how you display the image (FitToWindow? Fixed Size?).
        """
        # Example implementation for "Fit to Window" on a QLabel
        pixmap = self.image_label.pixmap()
        if not pixmap: return None
        
        img_w = pixmap.width()
        img_h = pixmap.height()
        widget_w = self.image_label.width()
        widget_h = self.image_label.height()
        
        # Calculate offset if aspect ratio is preserved (letterboxing)
        # This is complex, assuming simple stretch or perfect fit for this snippet
        # User needs to replace this with their actual view transform logic.
        
        # Pseudo-code for simple scaling (assuming image fills label)
        slide_w = self.wsi_handler.slide.dimensions[0]
        slide_h = self.wsi_handler.slide.dimensions[1]
        
        scale_x = slide_w / widget_w
        scale_y = slide_h / widget_h
        
        x1 = int(rect.left() * scale_x)
        y1 = int(rect.top() * scale_y)
        x2 = int(rect.right() * scale_x)
        y2 = int(rect.bottom() * scale_y)
        
        return (x1, y1, x2, y2)

    def show_batch_label_dialog(self, count):
        """Custom dialog to select label."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Annotation")
        layout = QVBoxLayout()
        
        lbl = QLabel(f"Selected {count} cells.\nAssign label:")
        layout.addWidget(lbl)
        
        btn_group = QButtonGroup(dialog)
        r1 = QRadioButton("Tumor (Positive)")
        r2 = QRadioButton("Normal (Negative)")
        r3 = QRadioButton("Unlabeled (Reset)")
        
        btn_group.addButton(r1, 1) # Label 1
        btn_group.addButton(r2, 0) # Label 0
        btn_group.addButton(r3, -1) # Reset
        r1.setChecked(True)
        
        layout.addWidget(r1)
        layout.addWidget(r2)
        layout.addWidget(r3)
        
        btns = QHBoxLayout()
        ok_btn = QPushButton("Confirm")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)
        
        dialog.setLayout(layout)
        
        if dialog.exec():
            return btn_group.checkedId(), True
        return None, False
