"""
preprocessor.py
===============
Pipeline Position : Step 1 — first module, called before anything else
Input             : image_path (str), mode ("rgb" or "ir")
Output            : preprocessed BGR numpy array

Reused from       : panel_cell_detection.py (lines 18-25)
                    sharpness + bilateral filter logic extracted here
"""

import cv2
import numpy as np
import logging
from PIL import Image, ImageEnhance

log = logging.getLogger(__name__)


def preprocess(image_path: str, mode: str = "rgb") -> np.ndarray:
    """
    Preprocess a solar panel image before panel detection or damage analysis.

    Args:
        image_path : path to image file (str)
        mode       : "rgb" or "ir"

    Returns:
        Preprocessed BGR numpy array

    Future integration:
        pipeline.py will call this as Step 1 for both RGB and IR paths
    """

    if mode not in ("rgb", "ir"):
        raise ValueError(f"mode must be 'rgb' or 'ir', got '{mode}'")

    # ── Load ──────────────────────────────────────────────────
    img_pil = Image.open(image_path).convert("RGB")
    log.info(f"[Preprocessor] Loaded {mode.upper()} | {image_path} | size={img_pil.size}")

    # ── RGB: sharpness + contrast enhance ─────────────────────
    # Reused from panel_cell_detection.py lines 18-21
    if mode == "rgb":
        img_pil = ImageEnhance.Sharpness(img_pil).enhance(2.0)
        img_pil = ImageEnhance.Contrast(img_pil).enhance(1.2)

    # ── Convert PIL → BGR (OpenCV format) ─────────────────────
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # ── Bilateral filter — edge-preserving denoise ─────────────
    # Reused from panel_cell_detection.py line 25
    # RGB: lighter | IR: stronger to smooth thermal noise
    if mode == "rgb":
        img = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
    else:
        img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # ── IR only: CLAHE on V channel for thermal contrast ───────
    if mode == "ir":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    log.info(f"[Preprocessor] Done | shape={img.shape}")
    return img


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python preprocessor.py <image_path> <rgb|ir>")
        sys.exit(1)

    path, mode = sys.argv[1], sys.argv[2]
    result = preprocess(path, mode)
    print(f"Output shape : {result.shape}")
    print(f"Output dtype : {result.dtype}")
    print(f"Preprocessor : OK")