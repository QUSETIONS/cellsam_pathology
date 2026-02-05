import os
import sqlite3
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import base64
from io import BytesIO
import webbrowser
from datetime import datetime

class ReportEngine:
    def __init__(self, db_path, wsi_handler, output_dir="reports"):
        self.db_path = db_path
        self.wsi_handler = wsi_handler
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_heatmap(self, slide_width, slide_height, thumbnail_path="wsi_thumbnail.png", alpha=0.5):
        """
        Generates a heatmap overlay on the WSI thumbnail based on tumor probability.
        """
        print("Generating heatmap...")
        
        # 1. Load data from DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Assuming table 'cells' has columns: x, y, tumor_prob
        # If tumor_prob is None, treat as 0
        cursor.execute("SELECT x, y, tumor_prob FROM cells WHERE tumor_prob IS NOT NULL")
        data = cursor.fetchall()
        conn.close()

        if not data:
            print("No prediction data found for heatmap.")
            return None

        data = np.array(data)
        xs = data[:, 0]
        ys = data[:, 1]
        probs = data[:, 2]

        # 2. Load Thumbnail
        if os.path.exists(thumbnail_path):
            thumb_img = cv2.imread(thumbnail_path)
            thumb_img = cv2.cvtColor(thumb_img, cv2.COLOR_BGR2RGB)
        else:
            # Fallback if no thumbnail file, try to generate one from handler
            # This is a placeholder, assuming handler has this method from context
             # If not, create a blank one
            thumb_img = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
            print(f"Warning: Thumbnail {thumbnail_path} not found. Using blank canvas.")

        thumb_h, thumb_w = thumb_img.shape[:2]
        
        # 3. Create Grid
        # Scale coordinates to thumbnail size
        scale_x = thumb_w / slide_width
        scale_y = thumb_h / slide_height
        
        grid_x = (xs * scale_x).astype(int)
        grid_y = (ys * scale_y).astype(int)
        
        # Clip to boundaries
        grid_x = np.clip(grid_x, 0, thumb_w - 1)
        grid_y = np.clip(grid_y, 0, thumb_h - 1)

        # 4. Aggregate Probabilities into a Heatmap Grid
        # We use a simple binning approach: sum of probabilities / count per bin (or just density)
        # Here we prioritize 'risk': average tumor probability in the region
        
        # Downsample grid for smoothness
        heatmap_resolution = (thumb_h // 16, thumb_w // 16) # Smaller grid for smoother map
        heatmap_acc = np.zeros(heatmap_resolution)
        heatmap_count = np.zeros(heatmap_resolution)
        
        # Map to smaller grid
        h_scale_x = heatmap_resolution[1] / thumb_w
        h_scale_y = heatmap_resolution[0] / thumb_h
        
        h_xs = (grid_x * h_scale_x).astype(int)
        h_ys = (grid_y * h_scale_y).astype(int)
        h_xs = np.clip(h_xs, 0, heatmap_resolution[1] - 1)
        h_ys = np.clip(h_ys, 0, heatmap_resolution[0] - 1)
        
        # Vectorized accumulation
        # Note: np.add.at is unbuffered in-place operation
        np.add.at(heatmap_acc, (h_ys, h_xs), probs)
        np.add.at(heatmap_count, (h_ys, h_xs), 1)
        
        # Avoid division by zero
        heatmap_avg = np.divide(heatmap_acc, heatmap_count, out=np.zeros_like(heatmap_acc), where=heatmap_count!=0)
        
        # 5. Generate Color Map
        # Resize back to thumbnail size
        heatmap_full = cv2.resize(heatmap_avg, (thumb_w, thumb_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap (Jet or Reds) using matplotlib to get proper RGBA
        # Normalize 0-1
        norm_heatmap = np.clip(heatmap_full, 0, 1)
        colormap = cm.get_cmap('jet')
        colored_heatmap = colormap(norm_heatmap) # Returns RGBA 0-1
        
        # Convert to 0-255 uint8
        colored_heatmap_uint8 = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
        
        # Create mask: where there are no cells, we want transparency or low opacity
        # We can use the 'count' resized to determine populated regions
        mask_map = cv2.resize(heatmap_count, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
        mask = mask_map > 0
        
        # Overlay
        # Result = Alpha * Heatmap + (1-Alpha) * Original
        # Only apply heatmap where mask is True (or low alpha elsewhere)
        
        overlay = thumb_img.copy()
        
        # Apply blending only on populated areas
        # We can also just blend everything but 0 probability areas will be blue (in jet) 
        # Ideally, 0 probability (background) should be transparent.
        
        # Better visualization: 
        # - Low prob: Blue/Transparent
        # - High prob: Red/Opaque
        
        overlay = cv2.addWeighted(colored_heatmap_uint8, alpha, thumb_img, 1 - alpha, 0)
        
        output_path = os.path.join(self.output_dir, "heatmap_overlay.png")
        # Convert RGB back to BGR for cv2 save (or use PIL)
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Heatmap saved to {output_path}")
        return output_path

    def export_final_report(self, heatmap_path, model_metrics=None):
        """
        Generates an HTML report.
        """
        print("Exporting report...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Stats
        cursor.execute("SELECT COUNT(*) FROM cells")
        total_cells = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM cells WHERE prediction = 1") # Assuming 1 is Tumor
        tumor_cells = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(tumor_prob) FROM cells WHERE prediction = 1")
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        conn.close()
        
        tumor_burden = (tumor_cells / total_cells * 100) if total_cells > 0 else 0
        
        # Diagnosis Logic
        diagnosis = "NORMAL / BENIGN"
        color = "green"
        if tumor_burden > 1.0: # Arbitrary threshold
            diagnosis = "POTENTIALLY MALIGNANT"
            color = "orange"
        if tumor_burden > 10.0:
            diagnosis = "HIGH PROBABILITY MALIGNANT"
            color = "red"

        # Encode image
        if heatmap_path and os.path.exists(heatmap_path):
            with open(heatmap_path, "rb") as img_file:
                b64_string = base64.b64encode(img_file.read()).decode('utf-8')
                img_tag = f'<img src="data:image/png;base64,{b64_string}" alt="Heatmap" style="width:100%; max-width:800px; border: 1px solid #ddd;"/>'
        else:
            img_tag = "<p>No heatmap available.</p>"

        # Simple HTML Template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pathology AI Diagnosis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }}
                h1, h2 {{ color: #2c3e50; }}
                .header {{ border-bottom: 2px solid #eee; padding-bottom: 20px; margin-bottom: 30px; }}
                .metric-box {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; display: inline-block; width: 45%; vertical-align: top; }}
                .diagnosis {{ font-size: 24px; font-weight: bold; color: {color}; padding: 15px; border: 2px solid {color}; border-radius: 5px; text-align: center; margin: 20px 0; }}
                .footer {{ margin-top: 50px; font-size: 12px; color: #888; border-top: 1px solid #eee; padding-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Pathology Assistant - Final Report</h1>
                <p>Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
            </div>

            <div class="diagnosis">
                {diagnosis}
            </div>

            <div class="metrics">
                <div class="metric-box">
                    <h3>Quantitative Metrics</h3>
                    <ul>
                        <li><strong>Total Cells Detected:</strong> {total_cells:,}</li>
                        <li><strong>Tumor Cells Detected:</strong> {tumor_cells:,}</li>
                        <li><strong>Tumor Burden:</strong> {tumor_burden:.2f}%</li>
                        <li><strong>Avg Confidence (Tumor):</strong> {avg_confidence:.2f}</li>
                    </ul>
                </div>
                <div class="metric-box">
                    <h3>Model Info</h3>
                    <p>Model: Random Forest (HITL Trained)</p>
                    <p>Status: {model_metrics if model_metrics else "Inference Complete"}</p>
                </div>
            </div>

            <h2>Whole Slide Heatmap</h2>
            <p>Visual distribution of predicted tumor probability (Jet Colormap: Blue=Low, Red=High).</p>
            {img_tag}

            <div class="footer">
                Generated by Gemini CLI Pathology Assistant. This report is for research use only and not for final clinical diagnosis.
            </div>
        </body>
        </html>
        """
        
        report_path = os.path.join(self.output_dir, "final_diagnosis.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        print(f"Report exported to {report_path}")
        return report_path
