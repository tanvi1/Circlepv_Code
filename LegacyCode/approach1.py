import cv2
import numpy as np
import os
from skimage.filters import frangi

# 1. Load the image
image_path = r'Images\SnailTrails_Microcracks1.png'  # Make sure this matches your image name/location
original = cv2.imread(image_path)
if original is None:
    raise FileNotFoundError(f"Image not found at {image_path}. Check your path.")

# Resize for uniform display if needed
h, w = original.shape[:2]
max_dim = 600
scale = max_dim / max(h, w)
original = cv2.resize(original, (int(w * scale), int(h * scale)))

# 2. Conversion to Grayscale
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# 3. Apply Frangi Filter (We do this ONCE outside the loop because it's computationally heavy)
gray_normalized = gray.astype(np.float32) / 255.0
print("Running Frangi Filter (this may take a few seconds)...")
frangi_img = frangi(gray_normalized, sigmas=range(1, 4, 1))
frangi_output = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# --- TEMPORARY FOLDER SETUP ---
ROOT_TEMP_DIR = "temp"
APPROACH_TEMP_DIR = os.path.join(ROOT_TEMP_DIR, "approach1")
os.makedirs(APPROACH_TEMP_DIR, exist_ok=True)
print(f"Temporary output directory created at: '{APPROACH_TEMP_DIR}'")

# --- UI & PIPELINE INITIALIZATION ---
window_name = "Approach 1: Frangi Filter Pipeline"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

# Global states for the UI
pipeline_frames = {}
stage_names = [
    "1_Original_Image", 
    "2_Grayscale_conversion", 
    "3_Frangi_Ridge_Detection", 
    "4_Binary_Thresholding", 
    "5_Morphological_Cleanup", 
    "6_Final_Polygon_Overlay"
]
current_stage_idx = 0
damage_percentage = 0.0

def process_pipeline(thresh_val):
    """
    Callback for the slider. Runs the thresholding, morphology, and area calculations.
    OpenCV blocks UI input while this runs, fulfilling your lock requirement.
    """
    global pipeline_frames, damage_percentage
    
    # 4. Binary Thresholding based on slider value
    _, binary = cv2.threshold(frangi_output, thresh_val, 255, cv2.THRESH_BINARY)

    # 5. Morphological Cleanup (Using CLOSE to bridge gaps in thin trails)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 6. Find Contours and Calculate Area
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out tiny noise contours (adjust area threshold 10 if needed)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
    
    # Calculate damage percentage strictly on the valid contours
    final_mask = np.zeros_like(cleaned_mask)
    cv2.drawContours(final_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
    
    total_pixels = original.shape[0] * original.shape[1]
    defect_pixels = cv2.countNonZero(final_mask)
    damage_percentage = (defect_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    # Draw Polygons on overlay
    polygon_overlay = original.copy()
    cv2.drawContours(polygon_overlay, valid_contours, -1, (0, 0, 255), 2)

    # Update global dictionary with new images
    pipeline_frames = {
        "1_Original_Image": original.copy(),
        "2_Grayscale_conversion": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        "3_Frangi_Ridge_Detection": cv2.cvtColor(frangi_output, cv2.COLOR_GRAY2BGR),
        "4_Binary_Thresholding": cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        "5_Morphological_Cleanup": cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR),
        "6_Final_Polygon_Overlay": polygon_overlay
    }
    
    # Overwrite images in the temp folder immediately
    for name, frame in pipeline_frames.items():
        cv2.imwrite(os.path.join(APPROACH_TEMP_DIR, f"{name}.jpg"), frame)

# Initialize Trackbar (Slider)
cv2.createTrackbar("Threshold", window_name, 15, 255, process_pipeline)

# Run the pipeline once with the initial value to populate frames
process_pipeline(15)

try:
    print("--------------------------------------------------")
    print("INSTRUCTIONS:")
    print("- Use the slider at the top to change the threshold.")
    print("- Press ANY KEY to cycle through the processing steps.")
    print("- Press 'ESC' to exit the window.")
    print("--------------------------------------------------")
    
    while True:
        # Get the frame for the current stage
        current_stage = stage_names[current_stage_idx]
        display_frame = pipeline_frames[current_stage].copy()
        
        # Add dynamic text (Stage Name and Damage %)
        cv2.putText(display_frame, current_stage.replace("_", " "), (10, display_frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(display_frame, f"Damage: {damage_percentage:.4f}%", (10, display_frame.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(window_name, display_frame)
        
        # Wait 50ms (keeps UI responsive to slider). Returns -1 if no key pressed.
        key = cv2.waitKey(50)
        
        if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            # ESC pressed or window closed manually
            break
        elif key != -1:  
            # Any other key pressed: cycle to the next image in the pipeline
            current_stage_idx = (current_stage_idx + 1) % len(stage_names)

finally:
    cv2.destroyAllWindows()
    print(f"\nExited. Temporary outputs remain in '{APPROACH_TEMP_DIR}'. Please delete manually when done.")