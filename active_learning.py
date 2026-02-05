import numpy as np
import pickle
import sqlite3
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class ActiveLearner:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ActiveLearner, cls).__new__(cls)
            cls._instance.model = make_pipeline(StandardScaler(), SGDClassifier(loss='log_loss', random_state=42))
            cls._instance.is_trained = False
        return cls._instance

    def train(self, X, y):
        """
        X: (N, D) features
        y: (N,) labels (0 or 1)
        """
        if len(np.unique(y)) < 2:
            print("Need at least one example of each class (Tumor/Normal) to train.")
            return False
            
        self.model.fit(X, y)
        self.is_trained = True
        return True

    def predict_proba(self, X):
        if not self.is_trained:
            return np.ones(len(X)) * 0.5 # Uncertainty
        return self.model.predict_proba(X)[:, 1] # Probability of class 1 (Tumor)

def train_on_labeled_cells(db_path):
    learner = ActiveLearner()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch labeled data: label != -1 (assuming -1 is unlabeled/default)
    cursor.execute("SELECT feature_data, label FROM cells WHERE label != -1 AND feature_data IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("No labeled data found.")
        return learner

    X = []
    y = []
    
    for blob, label in rows:
        try:
            feat = pickle.loads(blob)
            # Ensure feat is 1D array
            if hasattr(feat, 'flatten'): feat = feat.flatten()
            X.append(feat)
            y.append(label)
        except:
            continue
            
    if not X:
        return learner

    X = np.array(X)
    y = np.array(y)
    
    print(f"Training on {len(X)} samples...")
    success = learner.train(X, y)
    if success:
        print("Model updated.")
    
    return learner

def predict_unlabeled_cells(db_path, learner):
    if not learner.is_trained:
        return 0

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch UNLABELED data: label == -1
    cursor.execute("SELECT rowid, feature_data FROM cells WHERE label == -1 AND feature_data IS NOT NULL")
    rows = cursor.fetchall()
    
    if not rows:
        conn.close()
        return 0

    ids = []
    X = []
    
    for rowid, blob in rows:
        try:
            feat = pickle.loads(blob)
            if hasattr(feat, 'flatten'): feat = feat.flatten()
            ids.append(rowid)
            X.append(feat)
        except:
            continue
            
    if not X:
        conn.close()
        return 0
        
    X = np.array(X)
    probs = learner.predict_proba(X)
    predictions = (probs > 0.5).astype(int)
    
    # Batch update
    update_data = []
    for i, rowid in enumerate(ids):
        # Update prediction AND tumor_prob
        # prediction is the AI's best guess (0 or 1)
        # tumor_prob is the confidence (0.0 - 1.0)
        update_data.append((int(predictions[i]), float(probs[i]), rowid))
        
    cursor.executemany("UPDATE cells SET prediction = ?, tumor_prob = ? WHERE rowid = ?", update_data)
    conn.commit()
    conn.close()
    
    print(f"Propagated predictions to {len(update_data)} cells.")
    return len(update_data)
