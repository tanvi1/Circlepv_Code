from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO
import cv2
import json
import numpy as np
from preprocessor import preprocess


def detect_panels_yolo(img):
    # =====================================================
    # CONFIG
    # =====================================================

    ROBOFLOW_API_KEY = "3KMjy0miZ38nH6xSQ18z"

    YOLO_MODEL = r"bestl.pt"

    # =====================================================
    # LOAD YOLO
    # =====================================================

    model = YOLO(YOLO_MODEL)

    # =====================================================
    # ROBOFLOW CLIENT
    # =====================================================

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )

    # =====================================================
    # ARRAY DETECTION
    # =====================================================

    result = client.run_workflow(
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

    print(json.dumps(result, indent=2))

    # =====================================================
    # LOAD IMAGE
    # =====================================================


    if img is None:
        raise Exception("Image not found")

    final_img = img.copy()

    # =====================================================
    # FIND ARRAY BOXES
    # =====================================================

    boxes = []

    def find_boxes(data):

        if isinstance(data, dict):

            if (
                "x" in data and
                "y" in data and
                "width" in data and
                "height" in data
            ):
                boxes.append(data)

            for value in data.values():
                find_boxes(value)

        elif isinstance(data, list):

            for item in data:
                find_boxes(item)

    find_boxes(result)

    print(f"\nDetected Arrays: {len(boxes)}")

    # =====================================================
        # PROCESS EACH ARRAY
    # =====================================================

    for array_id, box in enumerate(boxes):

        try:

            cx = int(float(box["x"]))
            cy = int(float(box["y"]))
            w = int(float(box["width"]))
            h = int(float(box["height"]))

            x1 = max(0, cx - w // 2)
            y1 = max(0, cy - h // 2)

            x2 = min(img.shape[1], cx + w // 2)
            y2 = min(img.shape[0], cy + h // 2)

            # ----------------------------------
            # Draw array box
            # ----------------------------------

            cv2.rectangle(
                final_img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                final_img,
                f"Array {array_id+1}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            # ----------------------------------
            # Crop array
            # ----------------------------------

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # ----------------------------------
            # YOLO ON ARRAY ONLY
            # ----------------------------------

            results = model.predict(
                crop,
                conf=0.5,
                verbose=False,
                iou=0.5
            )

            r = results[0]

            names = model.names

            # ----------------------------------
            # Draw detections
            # ----------------------------------

            if r.boxes is not None:

                for det in r.boxes:

                    cls_id = int(det.cls[0])
                    conf = float(det.conf[0])

                    bx1, by1, bx2, by2 = det.xyxy[0].cpu().numpy()

                    bx1 = int(bx1)
                    by1 = int(by1)
                    bx2 = int(bx2)
                    by2 = int(by2)

                    # ----------------------------------
                    # Convert crop coords -> image coords
                    # ----------------------------------

                    gx1 = x1 + bx1
                    gy1 = y1 + by1
                    gx2 = x1 + bx2
                    gy2 = y1 + by2

                    label = f"{names[cls_id]} {conf:.2f}"

                    cv2.rectangle(
                        final_img,
                        (gx1, gy1),
                        (gx2, gy2),
                        (0, 0, 255),
                        2
                    )

            print(f"Array {array_id+1} processed")

        except Exception as e:
            print(f"Array {array_id+1} failed:", e)
    return final_img

img="C:\\Users\\riyasharma\\Documents\\Solar Project\\solar-panel-detection\\solar-panels\\5.jpg"
# img = preprocess(img)
img=cv2.imread(img)
final_img= detect_panels_yolo(img)

cv2.imshow("Final Result", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()