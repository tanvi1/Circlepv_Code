import cv2
import numpy as np
import os
import shutil

# 1. Load and prepare image
image_path = r'Images\SnailTrails_Microcracks1.png'
original = cv2.imread(image_path)
if original is None:
    raise FileNotFoundError("Image not found. Check your path.")
h, w = original.shape[:2]
scale = 600 / max(h, w)
original = cv2.resize(original, (int(w * scale), int(h * scale)))
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# 2. Fast Fourier Transform to move to frequency domain
dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Calculate Magnitude Spectrum for step-by-step visualization
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-5)
spectrum_vis = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 3. Construct a Notch/Band-Reject Filter Mask to block symmetrical grid points
rows, cols = gray.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols, 2), np.uint8)

# Creating vertical and horizontal band blocks to suppress structural grid patterns
r_width, c_width = int(rows * 0.05), int(cols * 0.05)
mask[crow - r_width:crow + r_width, :] = 0
mask[:, ccol - c_width:ccol + c_width] = 0
mask[crow - 5:crow + 5, ccol - 5:ccol + 5] = 1  # Preserve the core low-frequency lighting

# 4. Apply Mask and Inverse FFT
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Normalize reconstructed structural anomaly plane
anomaly_plane = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# Invert if backgrounds and lines flipped intensity values
anomaly_plane = cv2.threshold(anomaly_plane, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# 5. Extract Contours and Polygons
contours, _ = cv2.findContours(anomaly_plane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
polygon_overlay = original.copy()
cv2.drawContours(polygon_overlay, contours, -1, (0, 0, 255), 2)

pipeline_frames = {
    "1_Original_Grayscale": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
    "2_FFT_Frequency_Pattern": cv2.cvtColor(spectrum_vis, cv2.COLOR_GRAY2BGR),
    "3_Reconstructed_Anomalies": cv2.cvtColor(anomaly_plane, cv2.COLOR_GRAY2BGR),
    "4_Final_Anomaly_Polygons": polygon_overlay
}

# --- TEMPORARY FOLDER SETUP (WITH SUBFOLDER) ---
ROOT_TEMP_DIR = "temp"
os.makedirs(ROOT_TEMP_DIR, exist_ok=True)
APPROACH_TEMP_DIR = os.path.join(ROOT_TEMP_DIR, "approach3")  # Name of this code
os.makedirs(APPROACH_TEMP_DIR, exist_ok=True)
print(f"Temporary output directory: '{APPROACH_TEMP_DIR}'")

window_name = "Approach 3: Symmetry/FFT Anomaly Pipeline"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

try:
    print("Press any key to advance to the next step. Press 'ESC' to exit.")
    for current_stage, frame in pipeline_frames.items():
        display_frame = frame.copy()
        
        cv2.putText(display_frame, current_stage.replace("_", " "), (10, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
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