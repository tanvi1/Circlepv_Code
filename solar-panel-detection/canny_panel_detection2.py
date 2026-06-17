import cv2
import numpy as np
import os
from inference_sdk import InferenceHTTPClient
from preprocessor import preprocess
def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):

    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)

    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(
        rect,
        dst
    )

    return cv2.warpPerspective(
        image,
        M,
        (maxWidth, maxHeight)
    )
# =====================================================
# OUTPUT FOLDER
# =====================================================

os.makedirs("panels", exist_ok=True)

# =====================================================
# ROBOFLOW CONFIG
# =====================================================

ROBOFLOW_API_KEY = "3KMjy0miZ38nH6xSQ18z"

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)


def detect_panels_canny_roboflow(img):

    result = img.copy()

    # =====================================================
    # ARRAY DETECTION FROM ROBOFLOW
    # =====================================================

    rf_result = client.run_workflow(
        workspace_name="riyas-workspace-p2f1w",
        workflow_id="general-segmentation-api-2",
        images={
            "image": img
        },
        parameters={
            "classes": "Solar-Panels"
        },
        use_cache=True
    )

    # =====================================================
    # EXTRACT ARRAY BOXES
    # =====================================================

    array_boxes = []

    def find_boxes(data):

        if isinstance(data, dict):

            if (
                "x" in data and
                "y" in data and
                "width" in data and
                "height" in data
            ):
                array_boxes.append(data)

            for v in data.values():
                find_boxes(v)

        elif isinstance(data, list):

            for item in data:
                find_boxes(item)

    find_boxes(rf_result)

    print(f"Detected Arrays: {len(array_boxes)}")

    total_panels = 0
    array_count = 0
    panel_number = 0

    mask = np.zeros(
        (img.shape[0], img.shape[1]),
        dtype=np.uint8
    )

    # =====================================================
    # PROCESS EACH ARRAY
    # =====================================================

    for box in array_boxes:

        try:

            cx = int(float(box["x"]))
            cy = int(float(box["y"]))
            w = int(float(box["width"]))
            h = int(float(box["height"]))

            x = max(0, cx - w // 2)
            y = max(0, cy - h // 2)

            x2 = min(img.shape[1], cx + w // 2)
            y2 = min(img.shape[0], cy + h // 2)

            w = x2 - x
            h = y2 - y

            if w <= 0 or h <= 0:
                continue

            array_count += 1

            # ------------------------------------
            # DRAW ARRAY
            # ------------------------------------

            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
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

            cv2.rectangle(
                mask,
                (x, y),
                (x + w, y + h),
                255,
                -1
            )

            # ------------------------------------
            # ARRAY ROI
            # ------------------------------------

            roi = img[y:y+h, x:x+w]

            if roi.size == 0:
                continue

            # =====================================================
            # PANEL DETECTION USING YOUR CANNY LOGIC
            # =====================================================

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
            # FILTER PANELS
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
            # NMS
            # =====================================================

            if len(boxes) > 0:

                scores = [1.0] * len(boxes)

                indices = cv2.dnn.NMSBoxes(
                    boxes,
                    scores,
                    score_threshold=0.2,
                    nms_threshold=0.10
                )

                if len(indices) > 0:

                    kept_boxes = []

                    for idx in indices.flatten():

                        px, py, pw, ph = boxes[idx]

                        center_x = x + px + pw // 2
                        center_y = y + py + ph // 2

                        # Ensure center lies inside array box

                        if not (
                            x <= center_x <= x + w and
                            y <= center_y <= y + h
                        ):
                            continue

                        kept_boxes.append(
                            (px, py, pw, ph)
                        )

                    # =====================================================
                    # REMOVE CONTAINED BOXES
                    # =====================================================

                    final_boxes = []

                    for i, (px1, py1, pw1, ph1) in enumerate(kept_boxes):

                        contained = False

                        for j, (px2, py2, pw2, ph2) in enumerate(kept_boxes):

                            if i == j:
                                continue

                            if (
                                px1 >= px2 and
                                py1 >= py2 and
                                px1 + pw1 <= px2 + pw2 and
                                py1 + ph1 <= py2 + ph2
                            ):
                                contained = True
                                break

                        if not contained:
                            final_boxes.append(
                                (px1, py1, pw1, ph1)
                            )

                    # =====================================================
                    # DRAW PANELS + SAVE CROPS
                    # =====================================================

                    for (px, py, pw, ph) in final_boxes:

                        total_panels += 1
                        panel_number += 1

                        abs_x1 = x + px
                        abs_y1 = y + py

                        abs_x2 = abs_x1 + pw
                        abs_y2 = abs_y1 + ph

                        cv2.rectangle(
                            result,
                            (abs_x1, abs_y1),
                            (abs_x2, abs_y2),
                            (0, 255, 0),
                            2
                        )

                        label = str(panel_number)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.45
                        thickness = 1

                        (tw, th), _ = cv2.getTextSize(
                            label,
                            font,
                            font_scale,
                            thickness
                        )

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

                        panel_crop = img[
                            abs_y1:abs_y2,
                            abs_x1:abs_x2
                        ]

                        crop_filename = (
                            f"panels/array{array_count}_"
                            f"panel{panel_number:03d}.jpg"
                        )

                        cv2.imwrite(
                            crop_filename,
                            panel_crop
                        )

            print(f"Array {array_count} processed")

        except Exception as e:

            print(
                f"Array {array_count} failed:",
                e
            )

    # =====================================================
    # SUMMARY
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


# =====================================================
# MAIN
# =====================================================

img_path = r"C:\Users\riyasharma\Documents\Solar Project\solar-panel-detection\solar-panels\8.png"

img = preprocess(img_path)

result, mask, array_count, total_panels = detect_panels_canny_roboflow(img)

cv2.imwrite(
    "panels/result_annotated.jpg",
    result
)

print(
    f"Done! Arrays: {array_count} | "
    f"Panels saved: {total_panels}"
)

cv2.imshow("Array Mask", mask)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()