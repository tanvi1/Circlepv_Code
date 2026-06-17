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

def process_solar_shadows():
    # 1. Dynamically locate the Desktop folder
    desktop_path = get_onedrive_desktop_path()
    
    # 2. Look for solar_panel_test with any valid image extension directly on the Desktop
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    input_path = None
    matched_filename = None
    
    for ext in valid_extensions:
        filename = f"solar_panel_test{ext}"
        potential_path = os.path.join(desktop_path, filename)
        if os.path.exists(potential_path):
            input_path = potential_path
            matched_filename = filename
            break
            
    if not input_path:
        print(f"Error: Could not find 'solar_panel_test' image on Desktop.")
        return

    print(f"Processing test file: {input_path}")

    # 3. Read image
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Could not read the image file.")
        return

    output_img = img.copy()
    h, w, _ = img.shape
    
    # 4. Isolate the solar panels footprint broadly
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_footprint = np.array([90, 30, 20])
    upper_footprint = np.array([140, 255, 225])
    total_panel_mask = cv2.inRange(hsv, lower_footprint, upper_footprint)
    
    # Exclude bright white backing silicon lines from calculation boundaries
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, white_gaps = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    total_panel_mask = cv2.bitwise_and(total_panel_mask, cv2.bitwise_not(white_gaps))
    
    total_panel_pixels = cv2.countNonZero(total_panel_mask)
    if total_panel_pixels == 0:
        print("Error: Base panel installation area not detected.")
        return

    # 5. STRATEGY PART 1: Isolate the vertical Y-Axis shadow region via row profiling
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    row_means = np.zeros(h)
    for y in range(h):
        row_mask = total_panel_mask[y, :]
        if cv2.countNonZero(row_mask) > (w * 0.1): 
            # FIXED: Added index extraction to secure scalar float values from tuple
            row_means[y] = cv2.mean(blurred[y, :], mask=row_mask)[0]
            
    foreground_baseline = np.max(row_means)
    
    # Track the start and end row indices of the horizontal shadow block
    shadow_rows_mask = np.zeros_like(gray)
    for y in range(h):
        if row_means[y] > 0 and row_means[y] < (foreground_baseline * 0.65):
            shadow_rows_mask[y, :] = 255

    # 6. STRATEGY PART 2: Track where true, clean sunny color masks live on the X-axis
    lower_clean_sun = np.array([95, 100, 70])
    upper_clean_sun = np.array([135, 255, 255])
    clean_sun_mask = cv2.inRange(hsv, lower_clean_sun, upper_clean_sun)
    
    # Use Canny line patterns to solidify our map of unshaded panel objects
    edges = cv2.Canny(blurred, 40, 130)
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_pattern_mask = cv2.dilate(edges, kernel_line, iterations=1)
    
    # Combine colors and line textures to fully define the "Direct Sunlight Zones"
    clean_sunlight_zones = cv2.bitwise_or(clean_sun_mask, clean_pattern_mask)
    clean_sunlight_zones = cv2.bitwise_and(clean_sunlight_zones, total_panel_mask)

    # 7. HYBRID FILTER LINK: Intersect the shadow rows, keep panel footprint, slice out sunlit areas
    raw_shadow_mask = cv2.bitwise_and(shadow_rows_mask, total_panel_mask)
    raw_shadow_mask = cv2.bitwise_and(raw_shadow_mask, cv2.bitwise_not(clean_sunlight_zones))
    
    # CRUCIAL SPATIAL FILTER: Forcefully clear the overexposed bottom foreground area
    raw_shadow_mask[int(h * 0.80):h, :] = 0
    
    # Final morphological pass to completely close cell gaps within the shadow block
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    raw_shadow_mask = cv2.morphologyEx(raw_shadow_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Fill internal micro-holes in the shadow block before sorting components by size
    contours, _ = cv2.findContours(raw_shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_raw_mask = np.zeros_like(raw_shadow_mask)
    for cnt in contours:
        cv2.drawContours(filled_raw_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # 8. FINAL FILTER STEP: Connected Components Analysis to discard disjointed fragments
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_raw_mask, connectivity=8)
    
    solid_shadow_mask = np.zeros_like(filled_raw_mask)
    for i in range(1, num_labels):
        # FIXED: Corrected matrix dimension extraction syntax
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 3500:
            solid_shadow_mask[labels == i] = 255

    shadow_pixels = cv2.countNonZero(solid_shadow_mask)

    # 9. Calculate coverage metrics
    shadow_pct_of_panel = (shadow_pixels / total_panel_pixels) * 100
    shadow_pct_of_total = (shadow_pixels / (h * w)) * 100

    # 10. Visualize results (Paint the refined hybrid shadow mask bright red)
    # FIXED: Replaced open scalar assignment with explicit 3-channel BGR mask vector mapping
    output_img[solid_shadow_mask > 0] = [0, 0, 255]

    # 11. Save annotated image directly to the Desktop using the original filename
    output_path = os.path.join(desktop_path, matched_filename)
    cv2.imwrite(output_path, output_img)

    # 12. Print data summary
    print("\n--- Hybrid Spatial Localization Analysis Report ---")
    print(f"Total installation footprint pixels: {total_panel_pixels}")
    print(f"Verified shadow area pixels: {shadow_pixels}")
    print(f"Panel area obscured by shadow: {shadow_pct_of_panel:.2f}%")
    print(f"Total image obscured by shadow: {shadow_pct_of_total:.2f}%")
    print(f"Saved visual report to: {output_path}")

if __name__ == "__main__":
    process_solar_shadows()