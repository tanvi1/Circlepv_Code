import os
import sys
import cv2
import numpy as np

def get_onedrive_desktop_path():
    """Dynamically resolves the OneDrive or standard Desktop path on Windows."""
    home = os.path.expanduser("~")
    onedrive_commercial = os.path.join(home, "OneDrive - Commercial")
    onedrive_standard = os.path.join(home, "OneDrive")
    
    possible_paths = [
        os.path.join(onedrive_commercial, "Desktop"),
        os.path.join(onedrive_standard, "Desktop"),
        os.path.join(home, "Desktop"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    print("Error: Could not locate your Desktop directory automatically.")
    sys.exit(1)

def extract_gaussian_high_pass_cracks(img_gray, original_color_img):
    """Extracts all variable-width zigzag cracks and explicitly forces the capture of the white shattered center."""
    rows, cols = img_gray.shape[:2]
    img_output = original_color_img.copy()

    # STEP 1: DETECT DENSE WHITE SHATTERED CENTER CORE (Run on raw gray to avoid smoothing losses)
    # The center is a highly concentrated mix of white glass breaks and impact points
    _, raw_core_mask = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    
    # Close any small gaps inside the shattered center to form a solid core mask
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    raw_core_mask = cv2.morphologyEx(raw_core_mask, cv2.MORPH_CLOSE, core_kernel)
    
    core_contours, _ = cv2.findContours(raw_core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_core_mask = np.zeros_like(img_gray)
    
    img_center_y, img_center_x = rows // 2, cols // 2
    for cnt in core_contours:
        area = cv2.contourArea(cnt)
        # Target the main medium-to-large amorphous white central impact zone
        if 300 < area < 50000:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Exclude screenshot margins/borders
                x, y, w, h = cv2.boundingRect(cnt)
                if x <= 4 or y <= 4 or (x + w) >= cols - 4 or (y + h) >= rows - 4:
                    continue
                
                # Check proximity to center to ensure it's the primary impact blast site
                if abs(cx - img_center_x) < 350 and abs(cy - img_center_y) < 350:
                    cv2.drawContours(clean_core_mask, [cnt], -1, 255, -1)

    # STEP 2: FINE ZIGZAG LINE EXTRACTION
    smoothed = cv2.bilateralFilter(img_gray, d=9, sigmaColor=60, sigmaSpace=60)
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    bright_structures = cv2.morphologyEx(smoothed, cv2.MORPH_TOPHAT, tophat_kernel)

    # STEP 3: REFINED BACKGROUND GRID FILTERING
    # Isolate and subtract the strictly parallel vertical background stripes of the blue cell texture
    v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 12))
    vertical_grid = cv2.morphologyEx(bright_structures, cv2.MORPH_OPEN, v_structure)
    texture_free_cracks = cv2.subtract(bright_structures, vertical_grid)

    # Convert clean crack paths to a highly sensitive binary map
    canny = cv2.Canny(texture_free_cracks, 30, 90)
    combined = cv2.bitwise_or(texture_free_cracks, canny)
    _, binary_lines = cv2.threshold(combined, 30, 255, cv2.THRESH_BINARY)

    # STEP 4: FUSE ZIGZAG LINES AND THE WHITE SHATTERED CORE TOGETHER
    fused_cracks = cv2.bitwise_or(binary_lines, clean_core_mask)

    # STEP 5: FINAL GEOMETRIC CLEANUP
    contours, _ = cv2.findContours(fused_cracks, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    final_crack_mask = np.zeros_like(fused_cracks)
    
    crack_segments = 0
    has_dense_impact_core = np.count_nonzero(clean_core_mask) > 0

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, False)
        area = cv2.contourArea(cnt)
        
        if perimeter > 8:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h != 0 else 0
            
            # Wipe out any remaining screenshot edge noise
            if x <= 4 or y <= 4 or (x + w) >= cols - 4 or (y + h) >= rows - 4:
                continue

            # Reject perfectly horizontal main busbar panel lines
            if aspect_ratio > 12.0 and area < 200:
                continue

            cv2.drawContours(final_crack_mask, [cnt], -1, 255, -1)
            crack_segments += 1

    # STEP 6: Calculate precise quantitative damage metrics: (N_damage / N_panel) * 100
    n_damage = np.count_nonzero(final_crack_mask)
    n_panel = rows * cols
    damage_percentage = (n_damage / n_panel) * 100

    # STEP 7: Apply solid RED color annotation mask directly to the original color backdrop
    img_output[final_crack_mask > 0] = (0, 0, 255)

    # STEP 8: Overlay diagnostic text strings
    if has_dense_impact_core or crack_segments > 10:
        pattern_label = "Pattern: Shattered Spiderweb (Impact Center + Zigzag)"
    elif crack_segments > 3:
        pattern_label = "Pattern: Random Zigzag Crack Paths"
    else:
        pattern_label = "Pattern: Linear / Minor Fracture Line"
        
    damage_label = f"Damage Area: {damage_percentage:.2f}%"
    
    cv2.putText(img_output, pattern_label, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img_output, damage_label, (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    # STEP 9: Save result to OneDrive desktop
    desktop_dir = get_onedrive_desktop_path()
    output_filename = os.path.join(desktop_dir, "solar_final_diagnostic.jpg")
    cv2.imwrite(output_filename, img_output)
    
    return fused_cracks

# --- EXECUTION BLOCK ENGINE ---
if __name__ == "__main__":
    desktop = get_onedrive_desktop_path()
    possible_names = ["solar_panel_test.jpeg", "solar_panel_test.jpg", "solar_panel_test.png"]
    input_path = None
    
    for name in possible_names:
        test_path = os.path.join(desktop, name)
        if os.path.exists(test_path):
            input_path = test_path
            break
            
    if input_path is not None:
        orig = cv2.imread(input_path)
        if orig is not None:
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            extract_gaussian_high_pass_cracks(gray, orig)
