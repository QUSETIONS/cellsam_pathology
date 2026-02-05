from gui_virtual_cytometry import VirtualCytometryWindow

# --- 1. Integration in MainWindow.__init__ ---
# self.cytometry_window = None

# --- 2. Button Action to Open Window ---
def open_cytometry_view(self):
    if not self.cytometry_window:
        self.cytometry_window = VirtualCytometryWindow(self) # Make it non-modal or modal? Non-modal usually better for linking.
        self.cytometry_window.setWindowFlags(Qt.Window) # Independent window
        
        # Connect Signals
        self.cytometry_window.selection_changed.connect(self.on_cytometry_selection)
        self.cytometry_window.batch_label_requested.connect(self.on_cytometry_batch_label)
    
    # Prepare Data
    # 1. Get visible cells or sampled subset
    visible_cells = self.get_visible_cells() # From previous task
    if len(visible_cells) < 10:
        QMessageBox.information(self, "Info", "Not enough cells visible for analysis.")
        return

    # 2. Extract Embeddings
    # Assuming cells have 'embedding' (numpy array) and 'id'
    # If using DB, might need to fetch blobs.
    embeddings = []
    ids = []
    labels = []
    
    valid_cells = []
    for c in visible_cells:
        # Assuming your cell dict has 'embedding' loaded. 
        # If not, you might need to query DB here for these IDs.
        if 'embedding' in c and c['embedding'] is not None:
            embeddings.append(c['embedding'])
            ids.append(c['id'])
            labels.append(c.get('prediction', -1)) # or ground truth label
            valid_cells.append(c)
            
    if not embeddings:
         QMessageBox.warning(self, "Data Error", "No embeddings found. Upgrade to ConvNeXt pipeline first?")
         return

    import numpy as np
    embeddings_np = np.array(embeddings)
    
    # 3. Send to Window
    self.cytometry_window.set_data(embeddings_np, labels, ids)
    self.cytometry_window.show()
    self.cytometry_window.raise_()

# --- 3. Signal Handlers ---

def on_cytometry_selection(self, cell_ids):
    """
    Called when user selects points in Scatter Plot.
    Target: Highlight these cells in WSI View.
    """
    print(f"Highlighting {len(cell_ids)} cells from feature space.")
    
    # You need to implement a highlighting mechanism in your overlay
    # e.g., self.highlighted_ids = set(cell_ids)
    # self.update_overlay()
    
    self.highlighted_ids = set(cell_ids)
    self.update_overlay()

def on_cytometry_batch_label(self, cell_ids, new_label):
    """
    Called when user batch labels from Scatter Plot.
    """
    # 1. Update Database
    # cursor.execute(f"UPDATE cells SET label = ? ...")
    
    # 2. Retrain active learner
    # self.learner.teach(...)
    
    # 3. Refresh UI
    self.update_overlay()
    print(f"Batch labeled {len(cell_ids)} cells as {new_label}")
