import sqlite3
import numpy as np
import pickle

def predict_whole_slide(learner, db_path, batch_size=10000):
    """
    Extends the ActiveLearner to perform batch inference on the database.
    
    Args:
        learner: The trained ActiveLearner instance (must have .predict_proba() method).
        db_path: Path to sqlite database.
        batch_size: Number of cells to process at once to save memory.
    """
    print(f"Starting Whole Slide Inference on {db_path}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Check total count
    cursor.execute("SELECT COUNT(*) FROM cells")
    total_cells = cursor.fetchone()[0]
    print(f"Total cells to process: {total_cells}")
    
    if total_cells == 0:
        conn.close()
        return

    # 2. Iterate in batches
    # We select 'rowid' or 'id' to paginate safely, but LIMIT/OFFSET is easiest for now
    # Ideally, we should have an 'id' column. We will assume standard SQLite rowid if primary key is missing.
    
    processed = 0
    
    while processed < total_cells:
        # Fetch batch
        # Assuming we need to read 'features' column. 
        # If features are stored as BLOB (pickle or bytes), we need to deserialize.
        # If features are stored as individual columns, this query needs adjustment.
        # ADJUSTMENT: Assuming 'features' column contains bytes (e.g., from numpy.tobytes() or pickle)
        
        cursor.execute(f"SELECT rowid, feature_data FROM cells LIMIT {batch_size} OFFSET {processed}")
        rows = cursor.fetchall()
        
        if not rows:
            break
            
        row_ids = []
        feature_batch = []
        
        for r_id, f_blob in rows:
            row_ids.append(r_id)
            # Deserialize: Assuming pickle for generic object or numpy buffer
            # A common pattern is storing numpy arrays as bytes.
            # Here we try pickle first, or assume float32 array
            try:
                feat = pickle.loads(f_blob)
            except:
                # Fallback: assume flat float32 array
                feat = np.frombuffer(f_blob, dtype=np.float32)
            
            feature_batch.append(feat)
            
        X_batch = np.array(feature_batch)
        
        # Handle case where X_batch might be 1D if single feature? Unlikely.
        if X_batch.ndim == 1:
            X_batch = X_batch.reshape(-1, 1)

        # 3. Predict
        # learner.model should be the underlying sklearn estimator
        # learner.predict usually returns class, we want proba too if possible
        
        try:
            # If learner has sklearn model
            probs = learner.estimator.predict_proba(X_batch)
            # Binary classification: [prob_0, prob_1]
            # We want prob_1 (Tumor)
            tumor_probs = probs[:, 1]
            preds = (tumor_probs > 0.5).astype(int)
        except AttributeError:
            # Fallback if wrapper is different
            preds = learner.predict(X_batch)
            tumor_probs = preds # If only class returned, prob is 0 or 1
        
        # 4. Update Database
        # usage of executemany for speed
        update_data = list(zip(preds.tolist(), tumor_probs.tolist(), row_ids))
        
        cursor.executemany("UPDATE cells SET prediction = ?, tumor_prob = ? WHERE rowid = ?", update_data)
        conn.commit()
        
        processed += len(rows)
        print(f"Processed {processed}/{total_cells} cells...", end='\r')

    print("\nWhole Slide Inference Complete.")
    conn.close()
