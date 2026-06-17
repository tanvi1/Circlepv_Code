import cv2
import numpy as np
from skimage.filters import frangi

def run_detection_pipeline(image_path, window_idx):
    """
    This function acts as a wrapper. Every time we call it, it creates a 
    completely isolated environment (and window) for a single image.
    """
    # 1. Load the image
    original = cv2.imread(image_path)
    if original is None:
        print(f"Image not found. Check your path: {image_path}")
        return

    h, w = original.shape[:2]
    max_dim = 600
    scale = max_dim / max(h, w)
    original = cv2.resize(original, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Isolate bright lines (the grids and frames) against the dark panel
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -5)
        
    # Extract structural horizontal and vertical lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
        
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
        
    grid_mask = cv2.add(h_lines, v_lines)
    grid_line_width = 6 # Maximum normal width of a grid line in pixels
        
    # Dilate slightly to create an "overlap zone"
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (grid_line_width, grid_line_width))
    thick_grid_mask = cv2.dilate(grid_mask, thick_kernel, iterations=1)

    # ====================================================================
    # PHASE 1: FFT GRID REMOVAL
    # ====================================================================
    print(f"[{window_idx}] Running FFT to remove periodic grid structures...")
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create a Notch Filter Mask
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)

    # Block the main vertical and horizontal frequency spikes
    cross_width = 3 
    mask[crow - cross_width:crow + cross_width, :] = 0
    mask[:, ccol - cross_width:ccol + cross_width] = 0

    # Preserve the low-frequency DC component
    mask[crow - 15:crow + 15, ccol - 15:ccol + 15] = 1

    # Apply Mask and Inverse FFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize the FFT cleaned image
    fft_cleaned_normalized = cv2.normalize(img_back, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    fft_vis = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ====================================================================
    # PHASE 2: DUAL-POLARITY FRANGI ON CLEANED IMAGE
    # ====================================================================
    print(f"[{window_idx}] Running Dual-Polarity Frangi Filter...")
    frangi_dark = frangi(fft_cleaned_normalized, sigmas=range(1, 4, 1), black_ridges=True)
    frangi_light = frangi(fft_cleaned_normalized, sigmas=range(1, 4, 1), black_ridges=False)

    frangi_combined = np.maximum(frangi_dark, frangi_light)
    frangi_output = cv2.normalize(frangi_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ====================================================================
    # PHASE 3: INTERACTIVE THRESHOLDING & UI
    # ====================================================================
    window_name = f"Window {window_idx}: FFT + Frangi"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def process_pipeline(thresh_val):
        _, binary = cv2.threshold(frangi_output, thresh_val, 255, cv2.THRESH_BINARY)

        # Dynamic 2% Border Masking (Hardcoded to prevent UI variable errors)
        border_pct = 0.02
        bw_border = int(w * border_pct)
        bh_border = int(h * border_pct)
        
        binary[0:bh_border, :] = 0          
        binary[h-bh_border:h, :] = 0        
        binary[:, 0:bw_border] = 0          
        binary[:, w-bw_border:w] = 0        

        # Morphological Cleanup 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find Contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- OVERLAP, WIDTH & SHAPE FILTERING ---
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            
            # Slightly increased from 10 to 15 to drop microscopic noise
            if area > 15: 
                # 1. SHAPE FILTER (Kills the junction dots/circles)
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # If it's highly compact/solid (like a square pad or circle), reject it
                if solidity > 0.60:
                    continue
                    
                # 2. OVERLAP FILTER (Kills the long structural grid lines)
                c_mask = np.zeros_like(cleaned_mask)
                cv2.drawContours(c_mask, [c], -1, 255, thickness=cv2.FILLED)
                
                overlap = cv2.bitwise_and(c_mask, thick_grid_mask)
                overlap_area = cv2.countNonZero(overlap)
                
                if overlap_area > (0.5 * area):
                    rect = cv2.minAreaRect(c)
                    min_dim = min(rect[1][0], rect[1][1])
                    
                    if min_dim <= grid_line_width:
                        continue 
                
                valid_contours.append(c)
                
        # 4. Calculate Final Mask & Area
        final_mask = np.zeros_like(cleaned_mask)
        cv2.drawContours(final_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
        
        total_pixels = original.shape[0] * original.shape[1]
        defect_pixels = cv2.countNonZero(final_mask)
        damage_percentage = (defect_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        # Draw Overlays
        final_output = original.copy()
        cv2.drawContours(final_output, valid_contours, -1, (0, 0, 255), 2)
        cv2.putText(final_output, f"Damage Area: {damage_percentage:.4f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        combined_display = np.hstack((cv2.cvtColor(fft_vis, cv2.COLOR_GRAY2BGR), final_output))
        
        cv2.putText(combined_display, "FFT Cleaned (Grid Removed)", (10, combined_display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_display, "Final Hybrid Detection", (w + 10, combined_display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, combined_display)

    # Create the single trackbar and initialize for THIS window
    cv2.createTrackbar("Threshold", window_name, 35, 255, process_pipeline)
    process_pipeline(35)


# ====================================================================
# MAIN EXECUTION SCRIPT
# ====================================================================

# 1. Define your three images
image_paths = [
    r'Images\SnailTrailGen1.jpeg', 
    r'Images\SnailTrailGen2.jpeg', 
    r'Images\SnailTrailGen3.jpeg',
    r'Images\SnailTrails_Microcracks1.png'
]

# 2. Loop through and create a window for each image
for idx, path in enumerate(image_paths):
    run_detection_pipeline(path, idx + 1)

print("\nUse the sliders to adjust the threshold for each image independently.")
print("Press 'ESC' while clicked on any image window to exit the program.")

# 3. Keep the program running until you press Escape
while True:
    key = cv2.waitKey(50)
    # 27 is the Escape key
    if key == 27:
        break

cv2.destroyAllWindows()