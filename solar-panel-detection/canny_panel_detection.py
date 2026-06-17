import cv2
import numpy as np
from preprocessor import preprocess
import os

# =====================================================
# OUTPUT FOLDER
# =====================================================

os.makedirs("panels", exist_ok=True)
def detect_panels_canny(img):

    result = img.copy()

    # =====================================================
    # SOLAR ARRAY DETECTION
    # =====================================================

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([80, 20, 40])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

    kernel = np.ones((15, 15), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    array_contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    total_panels = 0
    array_count = 0
    panel_number = 0  # global panel counter across all arrays

    # =====================================================
    # PROCESS EACH SOLAR TABLE
    # =====================================================

    for cnt in array_contours:

        area = cv2.contourArea(cnt)

        if area < 5000 or area > 500000:
            continue

        peri = cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(
            cnt,
            0.02 * peri,
            True
        )

        x, y, w, h = cv2.boundingRect(approx)

        ar = w / float(h)

        if ar < 1 and ar> 1.5:
            continue

        array_count += 1

        cv2.drawContours(
            result,
            [approx],
            -1,
            (0, 255, 255),
            3
        )

        cv2.putText(
            result,
            f"Array {array_count}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # =====================================================
        # ROI
        # =====================================================

        roi = img[y:y+h, x:x+w]

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blur,
            30,
            250
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []

        # =====================================================
        # FILTERING
        # =====================================================

        for c in contours:

            px, py, pw, ph = cv2.boundingRect(c)

            panel_area = pw * ph

            if panel_area < 500:
                continue

            if panel_area > 20000:
                continue

            aspect_ratio = pw / float(ph)

            if aspect_ratio < 1.0:
                continue

            if aspect_ratio > 3.0:
                continue

            boxes.append([px, py, pw, ph])

        # =====================================================
        # REMOVE OVERLAPPING BOXES + OUTSIDE + CONTAINED
        # =====================================================

        if len(boxes) > 0:

            scores = [1.0] * len(boxes)

            indices = cv2.dnn.NMSBoxes(
                boxes,
                scores,
                score_threshold=0.1,
                nms_threshold=0.10
            )

            if len(indices) > 0:

                kept_boxes = []

                for idx in indices.flatten():
                    px, py, pw, ph = boxes[idx]

                    # Fix 1: skip boxes outside array contour
                    cx = x + px + pw // 2
                    cy = y + py + ph // 2
                    if cv2.pointPolygonTest(approx, (float(cx), float(cy)), False) < 0:
                        continue

                    kept_boxes.append((px, py, pw, ph))

                # Fix 2: remove completely contained (nested) boxes
                final_boxes = []
                for i, (px1, py1, pw1, ph1) in enumerate(kept_boxes):
                    contained = False
                    for j, (px2, py2, pw2, ph2) in enumerate(kept_boxes):
                        if i == j:
                            continue
                        if (px1 >= px2 and py1 >= py2 and
                                px1 + pw1 <= px2 + pw2 and
                                py1 + ph1 <= py2 + ph2):
                            contained = True
                            break
                    if not contained:
                        final_boxes.append((px1, py1, pw1, ph1))

                # =====================================================
                # DRAW + NUMBER + CROP EACH PANEL
                # =====================================================

                for (px, py, pw, ph) in final_boxes:

                    total_panels += 1
                    panel_number += 1

                    # Absolute coordinates on full image
                    abs_x1 = x + px
                    abs_y1 = y + py
                    abs_x2 = abs_x1 + pw
                    abs_y2 = abs_y1 + ph

                    # Draw green rectangle
                    cv2.rectangle(
                        result,
                        (abs_x1, abs_y1),
                        (abs_x2, abs_y2),
                        (0, 255, 0),
                        2
                    )

                    # Draw panel number (white text, black background for readability)
                    label = str(panel_number)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.45
                    thickness = 1
                    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

                    # Black filled rect behind text
                    cv2.rectangle(
                        result,
                        (abs_x1, abs_y1),
                        (abs_x1 + tw + 4, abs_y1 + th + 4),
                        (0, 0, 0),
                        -1
                    )

                    cv2.putText(
                        result,
                        label,
                        (abs_x1 + 2, abs_y1 + th + 1),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness
                    )

                    # Crop panel from original image and save
                    panel_crop = img[abs_y1:abs_y2, abs_x1:abs_x2]

                    crop_filename = f"panels/array{array_count}_panel{panel_number:03d}.jpg"
                    cv2.imwrite(crop_filename, panel_crop)

    # =====================================================
    # INFO
    # =====================================================

    cv2.putText(
        result,
        f"Arrays: {array_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        result,
        f"Panels: {total_panels}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )
    return result, mask, array_count, total_panels

img="C:\\Users\\riyasharma\\Documents\\Solar Project\\solar-panel-detection\\solar-panels\\7.png"
img = preprocess(img)
print(img)
result, mask, array_count, total_panels = detect_panels_canny(img)

# Save annotated result image too
cv2.imwrite("panels/result_annotated.jpg", result)
print(f"Done! Arrays: {array_count} | Panels saved: {total_panels}")
print(f"Crops saved in: panels/")

# =====================================================
# SHOW
# =====================================================

cv2.imshow("Mask", mask)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
