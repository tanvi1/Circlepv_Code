import os
import cv2
import numpy as np
from datetime import datetime

def strip_all_canvas_padding(raw_img):
    """Trims away background black/white canvas borders securely without tuple sorting bugs."""
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    canvas_mask = cv2.inRange(gray, 15, 240)
    contours, _ = cv2.findContours(canvas_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x_min, y_min = raw_img.shape[1], raw_img.shape[0]
        x_max, y_max = 0, 0
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            if w_box > 30 and h_box > 30: 
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w_box)
                y_max = max(y_max, y + h_box)
        if x_max > x_min and y_max > y_min:
            return raw_img[y_min:y_max, x_min:x_max]
    return raw_img.copy()

def detect_dust_across_full_frame(img_path, saturation_threshold=42, lightness_threshold=115, pattern_block=15):
    """
    Advanced structural visibility engine optimizing thick white occluded dust blocks 
    and thin brown/mud overcast coatings while ignoring background mesh zones and clean panels.
    """
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        raise FileNotFoundError(f"Could not open or find the image at: {img_path}")
        
    color_img = strip_all_canvas_padding(raw_img)
    annotated_img = color_img.copy()
    h, w, _ = color_img.shape
    n_frame_total = h * w
    
    gray_cropped = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    
    # Uniformly balance glare and transitions
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized_gray = clahe.apply(gray_cropped)
    
    # 1. PARALLEL GRID PATTERN IDENTIFIER (Extracting Busbars & Fingers)
    grad_x = cv2.Sobel(equalized_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(equalized_gray, cv2.CV_64F, 0, 1, ksize=3)
    pattern_magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Compute local variance of the pattern to determine sharpness/visibility
    mean_int = cv2.blur(pattern_magnitude, (pattern_block, pattern_block))
    mean_sq_int = cv2.blur(pattern_magnitude**2, (pattern_block, pattern_block))
    pattern_variance = np.sqrt(np.maximum(mean_sq_int - mean_int**2, 0))
    pattern_variance = np.clip(pattern_variance * 4, 0, 255).astype(np.uint8)

    # 2. COLOR SPACE PREPARATION
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 3. VISIBILITY SEGREGATION FILTERS (OPTIMIZED BYPASSING WATERMARK INTERFERENCE)
    
    # --- STRATEGY PROFILE 1: THICK DUST PATTERN (Complete Silicon Occlusion) ---
    # Targets matte, white-shaded coatings that hide the parallel fingers/busbars.
    # OPTIMIZATION: Relies directly on the high-lightness, low-saturation profile 
    # to bypass background watermark texture noise.
    thick_dust_mask = (v_ch > 125) & (v_ch < 235) & (s_ch < 45)

    # --- STRATEGY PROFILE 2: THIN DUST LAYER (Brown / Mud Overcast) ---
    # Targets the distinct brown/mud tone overcasted on top of the blue/black silicon.
    # OPTIMIZATION: Relaxed threshold parameters slightly to reliably capture thin soil tracks 
    # despite high background line details.
    thin_dust_mask = (h_ch >= 4) & (h_ch <= 34) & (s_ch > 10) & (s_ch < 75) & (v_ch > 60)

    # --- STRATEGY PROFILE 3: PRISTINE PANEL & SHADOW PROTECTION ---
    # Isolates clean, high-contrast downside panels catching shadow and deep blue/black gradients.
    clean_panel_protection = (h_ch >= 90) & (h_ch <= 140) & (s_ch >= 40) & (v_ch < 130)
    
    # Absolute background mesh and shadow floor limit to erase dark platform edges
    absolute_shadow_floor = (v_ch < 50)

    # 4. GLOBAL MATRIX FUSION (The Adaptive Union Layer)
    # Combine the thick white dust zones and the thin brown overcast layers
    fused_dust_map = cv2.bitwise_or(np.uint8(thick_dust_mask * 255), np.uint8(thin_dust_mask * 255))
    
    # CRITICAL OPTIMIZATION: Clear the protected clean panels and dark platform mesh interferences
    fused_dust_map[clean_panel_protection] = 0
    fused_dust_map[absolute_shadow_floor] = 0
    
    # Full frame active workspace bounds
    active_roi = np.zeros_like(fused_dust_map)
    active_roi[10:h-10, 10:w-10] = 255
    fused_dust_map = cv2.bitwise_and(fused_dust_map, active_roi)
    
    # Dynamic High-Pass White Line Subtraction: Erase crisp white metal frame tracks and structural rails
    _, sharp_white_mask = cv2.threshold(equalized_gray, 215, 255, cv2.THRESH_BINARY)
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_white_tracks = cv2.dilate(sharp_white_mask, thick_kernel, iterations=1)
    fused_dust_map[dilated_white_tracks == 255] = 0
    
    # Clean micro pixel fragments to form smooth continuous blocks around dust fields
    fused_dust_map = cv2.morphologyEx(fused_dust_map, cv2.MORPH_CLOSE, thick_kernel)
    fused_dust_map = cv2.morphologyEx(fused_dust_map, cv2.MORPH_OPEN, thick_kernel)
    
    # 5. CANNY BOUNDARY TRACING
    canny_edges = cv2.Canny(fused_dust_map, 30, 100)
    dilated_edges = cv2.dilate(canny_edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # Calculate overall frame dust pixel percentage relative to total frame size (N_dust / N_frame_total)
    total_dust_pixels = np.sum(fused_dust_map > 0)
    overall_frame_dust_percentage = (total_dust_pixels / n_frame_total) * 100 if n_frame_total > 0 else 0.0
    
    print("-" * 50)
    print(f"STRUCTURAL VISIBILITY COATINGS REPORT:")
    print(f"Total Frame Area Pixels (N_frame): {n_frame_total:,}")
    print(f"Detected Dusty Area Pixels (N_dust): {total_dust_pixels:,}")
    print(f"Overall Dust Coverage Percentage:   {overall_frame_dust_percentage:.1f}%")
    print("-" * 50)

    # 6. DRAW CONTOURS ENCLOSING THE DUST AREAS
    patch_contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for p_cnt in patch_contours:
        if 400 < cv2.contourArea(p_cnt) < (n_frame_total * 0.95):
            cv2.drawContours(annotated_img, [p_cnt], -1, (0, 0, 255), 1)

    # Global status text ribbon bar overlay at the top edge
    overlay = annotated_img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated_img, 0.4, 0, annotated_img)
    
    display_text = f"OVERALL PHOTO FRAME DUST COVERAGE: {overall_frame_dust_percentage:.1f}%"
    cv2.putText(annotated_img, display_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated_img, f"ANALYZED ({overall_frame_dust_percentage:.1f}%)"


if __name__ == "__main__":
    desktop_path = r"C:\Users\Tanvi\OneDrive\Desktop"
    input_image_path = os.path.join(desktop_path, "solar_panel_test.jpg") 
    output_image_path = os.path.join(desktop_path, "solar_final_diagnostic.jpg")
    log_file_path = os.path.join(desktop_path, "dust_log.txt")

    print(f"Executing Seamless Non-Visibility Array Scanner on: {input_image_path}")

    try:
        diagnostic_result, status_msg = detect_dust_across_full_frame(
            input_image_path, saturation_threshold=42, lightness_threshold=115, pattern_block=15
        )
        
        if cv2.imwrite(output_image_path, diagnostic_result):
            print(f"[SUCCESS] Pattern-visibility strategy complete! All dusty sectors updated.")
            print(f"Diagnostic layout stored cleanly at: {output_image_path}")
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file_path, "a") as log_file:
                log_file.write(f"[{timestamp}] Image: solar_panel_test.jpg | Diagnosis: {status_msg}\n")
        else:
            print(f"[ERROR] Export failed. Check target path folder properties.")
    except Exception as e:
        print(f"[ERROR] Pipeline execution stopped: {e}")
