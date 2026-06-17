import cv2
import numpy as np
from modules.RGB_Damage_area import DamageResult, detect_glass_break

def client_corrosion(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_brown, upper_brown = np.array([5, 20, 40]), np.array([25, 255, 220])
    color_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_points = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
    mask_display = np.zeros_like(cleaned_mask)
    annotated_img = img.copy()
    damage_percent = 0.0
    
    if len(all_points) > 0:
        global_points = np.vstack(all_points).squeeze()
        if len(global_points.shape) == 2:
            x, y, w, h = cv2.boundingRect(global_points)
            hull = cv2.convexHull(global_points)
            cv2.drawContours(mask_display, [hull], -1, 255, -1)
            
            box_mask = np.zeros_like(cleaned_mask)
            cv2.ellipse(box_mask, (int(x+w/2), int(y+h/2)), (int(w/1.1), int(h/1.1)), 0, 0, 360, 255, -1)
            mask_display = cv2.bitwise_and(mask_display, box_mask)
            
            final_contours, _ = cv2.findContours(mask_display, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if final_contours:
                cv2.drawContours(annotated_img, final_contours, -1, (0, 0, 255), 2)
            
            damage_percent = (cv2.countNonZero(mask_display) / (img.shape[0] * img.shape[1])) * 100

    return DamageResult(
        class_name="Corrosion_Discoloration_Delamination", detected=damage_percent > 0,
        confidence=0.85, damage_percentage=round(damage_percent, 2),
        damage_mask=mask_display, annotated_image=annotated_img
    )

def client_microcracks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output_img = img.copy()
    
    se_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, se_noise)
    text_less = cv2.absdiff(gray, background)
    
    thresh = cv2.adaptiveThreshold(text_less, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_h, img_w = img.shape[:2]
    has_damage = False
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if x < int(img_w * 0.22) or x > int(img_w * 0.88): continue
        if y > int(img_h * 0.82): continue
        if x <= 12 or y <= 12 or (x + w) >= (img_w - 12) or (y + h) >= (img_h - 12): continue
        if w > (img_w * 0.35) or h > (img_h * 0.35): continue
        if h > 1.2 * w and w < 16: continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = (4 * np.pi * area) / (perimeter ** 2)
            if compactness < 0.25 and (w > 20 or h > 10) and area > 40:
                cv2.drawContours(output_img, [contour], -1, (0, 0, 255), 2)
                has_damage = True

    damage_ratio = (1 / 3) * 100 if has_damage else 0.0  # Converted client's ratio to percentage
    return DamageResult(
        class_name="SnailTrails_Microcracks", detected=has_damage,
        confidence=0.80 if has_damage else 0.0, damage_percentage=round(damage_ratio, 2),
        annotated_image=output_img
    )

def client_shadow(img):
    output_img = img.copy()
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    lower_fp, upper_fp = np.array([90, 30, 20]), np.array([140, 255, 225])
    total_panel_mask = cv2.inRange(hsv, lower_fp, upper_fp)
    _, white_gaps = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    total_panel_mask = cv2.bitwise_and(total_panel_mask, cv2.bitwise_not(white_gaps))
    
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    row_means = np.zeros(h)
    for y in range(h):
        row_mask = total_panel_mask[y, :]
        if cv2.countNonZero(row_mask) > (w * 0.1): 
            row_means[y] = cv2.mean(blurred[y, :], mask=row_mask)[0]
            
    foreground_baseline = np.max(row_means) if len(row_means[row_means > 0]) > 0 else 1
    shadow_rows_mask = np.zeros_like(gray)
    for y in range(h):
        if 0 < row_means[y] < (foreground_baseline * 0.65):
            shadow_rows_mask[y, :] = 255

    clean_sun_mask = cv2.inRange(hsv, np.array([95, 100, 70]), np.array([135, 255, 255]))
    edges = cv2.Canny(blurred, 40, 130)
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_pattern_mask = cv2.dilate(edges, kernel_line, iterations=1)
    
    clean_sunlight_zones = cv2.bitwise_and(cv2.bitwise_or(clean_sun_mask, clean_pattern_mask), total_panel_mask)
    raw_shadow_mask = cv2.bitwise_and(cv2.bitwise_and(shadow_rows_mask, total_panel_mask), cv2.bitwise_not(clean_sunlight_zones))
    raw_shadow_mask[int(h * 0.80):h, :] = 0
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    raw_shadow_mask = cv2.morphologyEx(raw_shadow_mask, cv2.MORPH_CLOSE, kernel_close)
    
    contours, _ = cv2.findContours(raw_shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_raw_mask = np.zeros_like(raw_shadow_mask)
    for cnt in contours: cv2.drawContours(filled_raw_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled_raw_mask, connectivity=8)
    solid_shadow_mask = np.zeros_like(filled_raw_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 3500: solid_shadow_mask[labels == i] = 255

    total_panel_pixels = max(1, cv2.countNonZero(total_panel_mask))
    shadow_pixels = cv2.countNonZero(solid_shadow_mask)
    shadow_pct = (shadow_pixels / total_panel_pixels) * 100
    
    output_img[solid_shadow_mask > 0] = [0, 0, 255]

    return DamageResult(
        class_name="HardShading", detected=shadow_pct > 0,
        confidence=0.85, damage_percentage=round(shadow_pct, 2),
        damage_mask=solid_shadow_mask, annotated_image=output_img
    )

def client_capture_glass(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_output = img.copy()
    rows, cols = img_gray.shape[:2]

    _, raw_core_mask = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    raw_core_mask = cv2.morphologyEx(raw_core_mask, cv2.MORPH_CLOSE, core_kernel)
    
    core_contours, _ = cv2.findContours(raw_core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_core_mask = np.zeros_like(img_gray)
    img_center_y, img_center_x = rows // 2, cols // 2
    
    for cnt in core_contours:
        if 300 < cv2.contourArea(cnt) < 50000:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(cnt)
                if x <= 4 or y <= 4 or (x + w) >= cols - 4 or (y + h) >= rows - 4: continue
                if abs(cx - img_center_x) < 350 and abs(cy - img_center_y) < 350:
                    cv2.drawContours(clean_core_mask, [cnt], -1, 255, -1)

    smoothed = cv2.bilateralFilter(img_gray, d=9, sigmaColor=60, sigmaSpace=60)
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    bright_structures = cv2.morphologyEx(smoothed, cv2.MORPH_TOPHAT, tophat_kernel)

    v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 12))
    vertical_grid = cv2.morphologyEx(bright_structures, cv2.MORPH_OPEN, v_structure)
    texture_free_cracks = cv2.subtract(bright_structures, vertical_grid)

    canny = cv2.Canny(texture_free_cracks, 30, 90)
    combined = cv2.bitwise_or(texture_free_cracks, canny)
    _, binary_lines = cv2.threshold(combined, 30, 255, cv2.THRESH_BINARY)

    fused_cracks = cv2.bitwise_or(binary_lines, clean_core_mask)
    contours, _ = cv2.findContours(fused_cracks, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    final_crack_mask = np.zeros_like(fused_cracks)

    for cnt in contours:
        if cv2.arcLength(cnt, False) > 8:
            x, y, w, h = cv2.boundingRect(cnt)
            if x <= 4 or y <= 4 or (x + w) >= cols - 4 or (y + h) >= rows - 4: continue
            if (float(w) / max(1, h)) > 12.0 and cv2.contourArea(cnt) < 200: continue
            cv2.drawContours(final_crack_mask, [cnt], -1, 255, -1)

    damage_percentage = (np.count_nonzero(final_crack_mask) / (rows * cols)) * 100
    img_output[final_crack_mask > 0] = (0, 0, 255)

    return DamageResult(
        class_name="GlassBreak", detected=damage_percentage > 0,
        confidence=0.85, damage_percentage=round(damage_percentage, 2),
        damage_mask=final_crack_mask, annotated_image=img_output
    )

def client_dust(img):
    """Extracted from client's solar_dust.py"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    thick_dust_mask = (v_ch > 125) & (v_ch < 235) & (s_ch < 45)
    thin_dust_mask = (h_ch >= 4) & (h_ch <= 34) & (s_ch > 10) & (s_ch < 75) & (v_ch > 60)
    clean_panel_protection = (h_ch >= 90) & (h_ch <= 140) & (s_ch >= 40) & (v_ch < 130)
    
    fused_dust_map = cv2.bitwise_or(np.uint8(thick_dust_mask * 255), np.uint8(thin_dust_mask * 255))
    fused_dust_map[clean_panel_protection] = 0
    fused_dust_map[v_ch < 50] = 0
    
    total_pixels = img.shape[0] * img.shape[1]
    dust_pct = (np.count_nonzero(fused_dust_map) / total_pixels) * 100
    
    output_img = img.copy()
    output_img[fused_dust_map > 0] = (0, 0, 255)
    
    return DamageResult(
        class_name="SoftShading_Soiling", detected=dust_pct > 0,
        confidence=0.85, damage_percentage=round(dust_pct, 2),
        damage_mask=fused_dust_map, annotated_image=output_img
    )

def client_snail_trail_canny(img):
    """Requested Canny Edge implementation for Snail Trails"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

def client_combined_snail_microcrack(img):
    """Merges Client's Microcrack with the new Canny Snail Trail"""
    res_micro = client_microcracks(img)
    canny_mask = client_snail_trail_canny(img)
    
    mask1 = res_micro.damage_mask if res_micro.damage_mask is not None else np.zeros(img.shape[:2], dtype=np.uint8)
    fused_mask = cv2.bitwise_or(mask1, canny_mask)
    
    dmg_pct = (np.count_nonzero(fused_mask) / (img.shape[0] * img.shape[1])) * 100
    output_img = img.copy()
    output_img[fused_mask > 0] = (0, 0, 255)
    
    return DamageResult(
        class_name="SnailTrails_Microcracks", detected=dmg_pct > 0,
        confidence=max(res_micro.confidence, 0.80 if dmg_pct > 0 else 0.0), 
        damage_percentage=round(dmg_pct, 2),
        damage_mask=fused_mask, annotated_image=output_img
    )

def client_merged_glass_break(img):
    """Merges Client's Without-Spider with the Default Spider logic"""
    res_capture = client_capture_glass(img)
    res_spider = detect_glass_break(img)
    
    mask1 = res_capture.damage_mask if res_capture.damage_mask is not None else np.zeros(img.shape[:2], dtype=np.uint8)
    mask2 = res_spider.damage_mask if res_spider.damage_mask is not None else np.zeros(img.shape[:2], dtype=np.uint8)
    fused_mask = cv2.bitwise_or(mask1, mask2)
    
    dmg_pct = (np.count_nonzero(fused_mask) / (img.shape[0] * img.shape[1])) * 100
    output_img = img.copy()
    output_img[fused_mask > 0] = (0, 0, 255)
    
    return DamageResult(
        class_name="GlassBreak", detected=dmg_pct > 0,
        confidence=max(res_capture.confidence, res_spider.confidence), 
        damage_percentage=round(dmg_pct, 2),
        damage_mask=fused_mask, annotated_image=output_img
    )

def run_client_module(image: np.ndarray, class_name: str):
    """Router matching YOLO categories to the client's custom scripts."""
    routing = {
        "EVA Delamination / Discoloration / Corrosion": client_corrosion,
        "Corrosion_Discoloration_Delamination": client_corrosion,
        "SnailTrails_Microcracks": client_combined_snail_microcrack,
        "Snail Trail / Microcrack": client_combined_snail_microcrack,
        "HardShading": client_shadow,
        "Hot Spots – Soiling/Shading (Hard, Bird Poop)": client_dust,
        "Soft Shading / Soiling": client_dust,
        "SoftShading_Soiling": client_dust,
        "GlassBreak": client_merged_glass_break,
        "Glass Break": client_merged_glass_break
    }
    
    if class_name in routing:
        return routing[class_name](image)
    return None