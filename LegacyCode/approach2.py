import cv2
import numpy as np
import os
import shutil
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# 1. Load Image
image_path = r'Images\SnailTrails_Microcracks1.png'
original = cv2.imread(image_path)
if original is None:
    raise FileNotFoundError("Image not found. Check your path.")
h, w = original.shape[:2]
scale = 600 / max(h, w)
original = cv2.resize(original, (int(w * scale), int(h * scale)))
rgb_img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# 2. Initialize SAM (Using ViT-B model for speed)
print("Loading SAM Model weights...")
sam_checkpoint = "sam_vit_b_01ec64.pth"  # Download this file from Meta's repo
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)

print("Generating Masks via Zero-Shot Inference...")
masks = mask_generator.generate(rgb_img)

# 3. Create Intermediate Visualizations
all_masks_img = original.copy()
filtered_masks_img = original.copy()

total_pixels = original.shape[0] * original.shape[1]
defect_pixel_count = 0

# Process generated masks
for m in masks:
    mask_poly = m['segmentation'].astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_poly, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw all detected shapes in green
    cv2.drawContours(all_masks_img, contours, -1, (0, 255, 0), 1)
    
    # FILTER ANOMALIES
    bbox = m['bbox']  # [x, y, width, height]
    aspect_ratio = max(bbox[2], bbox[3]) / (min(bbox[2], bbox[3]) + 1e-5)
    
    if aspect_ratio > 2.5 or (m['area'] < (total_pixels * 0.05) and m['stability_score'] > 0.85):
        cv2.drawContours(filtered_masks_img, contours, -1, (0, 0, 255), 2)
        defect_pixel_count += m['area']

damage_percentage = (defect_pixel_count / total_pixels) * 100
cv2.putText(filtered_masks_img, f"Damage: {damage_percentage:.2f}%", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

pipeline_frames = {
    "1_Original_Input_Image": original,
    "2_SAM_Native_Object_Masking": all_masks_img,
    "3_Filtered_Anomalies": filtered_masks_img
}

# --- TEMPORARY FOLDER SETUP (WITH SUBFOLDER) ---
ROOT_TEMP_DIR = "temp"
os.makedirs(ROOT_TEMP_DIR, exist_ok=True)
APPROACH_TEMP_DIR = os.path.join(ROOT_TEMP_DIR, "approach2")  # Name of this code
os.makedirs(APPROACH_TEMP_DIR, exist_ok=True)
print(f"Temporary output directory: '{APPROACH_TEMP_DIR}'")

window_name = "Approach 2: SAM Zero-Shot Pipeline"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

try:
    print("Press any key to advance to the next step. Press 'ESC' to exit.")
    for current_stage, frame in pipeline_frames.items():
        display_frame = frame.copy()
        
        cv2.putText(display_frame, current_stage.replace("_", " "), (10, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Save the frame to the unique subfolder
        save_path = os.path.join(APPROACH_TEMP_DIR, f"{current_stage}.jpg")
        cv2.imwrite(save_path, display_frame)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(0)
        if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    cv2.destroyAllWindows()
    # --- CLEANUP (User requested to manually delete) ---
    # if os.path.exists(ROOT_TEMP_DIR):
    #     shutil.rmtree(ROOT_TEMP_DIR)
    #     print(f"Cleaned up and deleted '{ROOT_TEMP_DIR}' directory.")
    print(f"Temporary outputs remain in '{APPROACH_TEMP_DIR}'. Please delete manually when done.")