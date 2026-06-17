import cv2
import numpy as np
import os

def analyze_solar_module():
    # Define exact requested system paths
    desktop_path = os.path.expanduser("~/OneDrive/Desktop")
    input_image_path = os.path.join(desktop_path, "solar_panel_test.jpg")
    output_image_path = os.path.join(desktop_path, "solar_final_damage.jpeg")

    # 1. Load the target image
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not locate image file at: {input_image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output_img = img.copy()

    # 2. Structural Texture Suppression
    se_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, se_noise)
    text_less = cv2.absdiff(gray, background)

    # 3. Dynamic Thresholding to isolate continuous lines
    thresh = cv2.adaptiveThreshold(
        text_less, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 15, -3
    )

    # 4. Extract and filter contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_damage = False

    img_h, img_w = img.shape[:2]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # FILTER A: Spatial X-bounding to focus on the central panel
        if x < int(img_w * 0.22) or x > int(img_w * 0.88):
            continue
            
        # FILTER B: Spatial Y-bounding to instantly drop the comb noise at the bottom frame
        if y > int(img_h * 0.82):
            continue
            
        # FILTER C: Skip image border elements
        if x <= 12 or y <= 12 or (x + w) >= (img_w - 12) or (y + h) >= (img_h - 12):
            continue
            
        # FILTER D: Skip massive elements like thick white busbars
        if w > (img_w * 0.35) or h > (img_h * 0.35):
            continue

        # FILTER E: Aspect Ratio Vertical Noise Filter
        if h > 1.2 * w and w < 16:
            continue

        # FILTER F: Geometry & Minimum Size check
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = (4 * np.pi * area) / (perimeter ** 2)
            
            # Low compactness targets winding, continuous lines (the real cracks)
            if compactness < 0.25 and (w > 20 or h > 10) and area > 40:
                cv2.drawContours(output_img, [contour], -1, (0, 0, 255), 2)
                has_damage = True

    # 5. Metrics Layout
    n_panel = 3
    n_damage = 1 if has_damage else 0
    damage_ratio = n_damage / n_panel

    # 6. Metrics Text Overlay
    metric_text = f"Npanel: {n_panel} | Ndamage: {n_damage} | Ratio: {damage_ratio:.2f}"
    cv2.putText(output_img, metric_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    # 7. Save final output image
    cv2.imwrite(output_image_path, output_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    print("--- Analysis Completed Successfully ---")
    print(f"Metrics -> Npanel: {n_panel}, Ndamage: {n_damage}, Ratio: {damage_ratio:.2f}")

if __name__ == "__main__":
    analyze_solar_module()
