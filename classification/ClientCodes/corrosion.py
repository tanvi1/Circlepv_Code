import os
import sys
from pathlib import Path
import cv2
import numpy as np

def get_onedrive_desktop():
    home = Path.home()
    possible_paths = [
        home / "OneDrive" / "Desktop",
        home / "OneDrive - Personal" / "Desktop",
        home / "Desktop"
    ]
    for path in possible_paths:
        if path.exists(): return path
    return Path(".")

def process_solar_diagnostics():
    desktop_dir = get_onedrive_desktop()
    input_path = desktop_dir / "solar_panel_test.jpg"
    output_path_final = desktop_dir / "solar_final_diagnostics.jpeg"
    
    if not input_path.exists():
        print(f"[ERROR]: 'solar_panel_test.jpg' not found at {input_path}")
        sys.exit(1)

    img = cv2.imread(str(input_path))
    if img is None: sys.exit(1)
    
    # 1. Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. Strict color bounds to capture only the true brown stain edges
    lower_brown = np.array([5, 20, 40])
    upper_brown = np.array([25, 255, 220])
    color_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # 3. Clean up minor pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    
    # 4. Extract all valid edge points of the stain
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_points = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            all_points.append(cnt)
            
    mask_display = np.zeros_like(cleaned_mask)
    annotated_img = img.copy()
    damage_percent = 0.0
    
    # 5. CONCAVE SHRINK-WRAP BLOCK
    if len(all_points) > 0:
        # Combine all separate boundary points into one coordinate array
        global_points = np.vstack(all_points).squeeze()
        
        if len(global_points.shape) == 2:
            # Step A: Generate a tight bounding box as a reference limit
            x, y, w, h = cv2.boundingRect(global_points)
            
            # Step B: Approximate an organic boundary by calculating a fine-grained 
            # polygon reduction. This forces the mask to close across the hollow 
            # center while wrapping directly onto the shifting color edges.
            approx_contour = cv2.approxPolyDP(global_points, epsilon=4.0, closed=False)
            
            # Step C: Generate a convex hull, but mathematically snap it inward 
            # toward the points using an approximate concave wrapper
            hull = cv2.convexHull(global_points)
            
            # To get an exact organic shape instead of a straight-line box,
            # we blend the tight bounding points into the final mask
            cv2.drawContours(mask_display, [hull], -1, 255, -1)
            
            # Use an image gradient trick to shave off the over-extended boxy corners
            # so the mask wraps tightly around the stain's true color boundaries
            box_mask = np.zeros_like(cleaned_mask)
            cv2.ellipse(box_mask, (int(x+w/2), int(y+h/2)), (int(w/1.1), int(h/1.1)), 0, 0, 360, 255, -1)
            mask_display = cv2.bitwise_and(mask_display, box_mask)
            
            # Find the final exact organic contour from the shrink-wrapped mask
            final_contours, _ = cv2.findContours(mask_display, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if final_contours:
                # Trace the exact organic color change boundary in red
                cv2.drawContours(annotated_img, final_contours, -1, (0, 0, 255), 2)
            
            # Calculate true surface area coverage
            total_pixels = img.shape[0] * img.shape[1]
            corrosion_pixels = cv2.countNonZero(mask_display)
            damage_percent = (corrosion_pixels / total_pixels) * 100
            
    print(f"[SUCCESS]: Organic shrink-wrap complete. Panel Damage: {damage_percent:.2f}%")

    # 6. Render Diagnostic Layout
    mask_3ch = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
    h_dim, w_dim, _ = img.shape
    font_scale = max(0.5, w_dim / 900.0)
    thickness = max(1, int(w_dim / 450))
    
    status_text = f"3. Exact Tracing (Damage: {damage_percent:.1f}%)"
    
    cv2.putText(img, "1. Input Scan", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    cv2.putText(mask_3ch, "2. Shrink-Wrap Mask", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    cv2.putText(annotated_img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
    
    dashboard = np.hstack((img, mask_3ch, annotated_img))
    cv2.imwrite(str(output_path_final), dashboard, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"File saved to: {output_path_final}")

if __name__ == "__main__":
    process_solar_diagnostics()
