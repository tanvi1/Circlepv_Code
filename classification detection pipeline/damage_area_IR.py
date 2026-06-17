"""
damage_area_IR.py
=================
IR Damage Area Analysis  —  Step 4a of pipeline

Input  : clf_result (classify_panel() output)  +  ir_array (np.ndarray)
Output : clf_result extended with IR damage fields

Does NOT do:
  - YOLO inference             → classifier.py
  - RGB damage area            → damage_area_RGB.py
  - Environmental corrections  → env_correction.py
  - Severity / final fusion    → severity_fusion.py

Usage
-----
>>> ir_array   = cv2.imread("panel_ir.png")          # BGR, pseudocolor
>>> clf_result = classify_panel(1, "ir", ir_array=ir_array, tmin=20, tmax=100)
>>> result     = analyze_ir_damage(clf_result, ir_array)

Output dict adds these keys to clf_result:
{
    "ir_healthy_avg_intensity" : 142.3,
    "ir_hotspot_avg_intensity" : 210.7,
    "delta_t_proxy"            : 68.4,
    "ir_damage_percent"        : 12.5,
    "ir_hotspot_mask"          : np.ndarray,   # binary mask → severity_fusion.py
    "ir_analysis_status"       : "ok",         # "ok" | "no_hotspot" | "error"
}
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# HSV THRESHOLDS  (pseudocolor / ironbow / rainbow IR)
# hot → red / orange / yellow / white
# ──────────────────────────────────────────────────────────────────────────────

_HSV_HOT_RANGES = [
    (np.array([0,   80,  80]),  np.array([10,  255, 255])),   # red low
    (np.array([170, 80,  80]),  np.array([180, 255, 255])),   # red high
    (np.array([10,  80,  80]),  np.array([25,  255, 255])),   # orange
    (np.array([20,  80,  80]),  np.array([40,  255, 255])),   # yellow
    (np.array([0,   0,   180]), np.array([180, 80,  255])),   # white
]

_MORPH_K   = 5    # open + close kernel size
_DILATE_K  = 15   # expand hotspot blobs
_MIN_AREA  = 100  # ignore tiny noise contours (pixels)


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _build_hotspot_mask(bgr: np.ndarray) -> np.ndarray:
    """HSV threshold → morphological cleanup → final binary mask."""
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lo, hi in _HSV_HOT_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    k1       = np.ones((_MORPH_K,  _MORPH_K),  np.uint8)
    k2       = np.ones((_DILATE_K, _DILATE_K), np.uint8)
    mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k1)
    mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
    expanded = cv2.dilate(mask, k2, iterations=1)

    contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros_like(expanded)
    for cnt in contours:
        if cv2.contourArea(cnt) > _MIN_AREA:
            cv2.drawContours(final, [cnt], -1, 255, cv2.FILLED)

    return final


def _compute_stats(bgr: np.ndarray, hotspot_mask: np.ndarray) -> dict:
    """Grayscale intensity stats — proxy for temperature on pseudocolor IR."""
    gray         = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    healthy_mask = cv2.bitwise_not(hotspot_mask)

    hot_px     = gray[hotspot_mask  > 0]
    healthy_px = gray[healthy_mask  > 0]

    t_hot  = float(np.mean(hot_px))     if len(hot_px)     > 0 else 0.0
    t_avg  = float(np.mean(healthy_px)) if len(healthy_px) > 0 else 0.0
    delta  = t_hot - t_avg

    n_hot   = int(np.count_nonzero(hotspot_mask))
    n_panel = int(hotspot_mask.shape[0] * hotspot_mask.shape[1])
    dmg_pct = (n_hot / n_panel * 100.0) if n_panel > 0 else 0.0

    return {
        "ir_healthy_avg_intensity": round(t_avg,   2),
        "ir_hotspot_avg_intensity": round(t_hot,   2),
        "delta_t_proxy":            round(delta,   2),
        "ir_damage_percent":        round(dmg_pct, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def analyze_ir_damage(
    clf_result: dict,
    ir_array:   np.ndarray,
) -> dict:
    """
    Extend classifier output with IR damage area stats.

    Parameters
    ----------
    clf_result : dict        — direct output of classify_panel()
                               (arrays NOT expected inside this dict)
    ir_array   : np.ndarray — same IR panel crop that was passed to
                               classify_panel() — BGR uint8, pseudocolor

    Returns
    -------
    dict — all clf_result keys  +  IR damage keys (see module docstring)
    """
    result = dict(clf_result)   # shallow copy, don't mutate original

    if ir_array is None:
        logger.warning("ir_array is None — skipping IR damage analysis")
        result.update({
            "ir_healthy_avg_intensity": None,
            "ir_hotspot_avg_intensity": None,
            "delta_t_proxy":            None,
            "ir_damage_percent":        None,
            "ir_hotspot_mask":          None,
            "ir_analysis_status":       "error",
        })
        return result

    try:
        # ensure BGR
        bgr = ir_array if (ir_array.ndim == 3 and ir_array.shape[2] == 3) \
              else cv2.cvtColor(ir_array, cv2.COLOR_GRAY2BGR)

        hotspot_mask = _build_hotspot_mask(bgr)
        stats        = _compute_stats(bgr, hotspot_mask)
        status       = "ok" if np.count_nonzero(hotspot_mask) > 0 else "no_hotspot"

        result.update({
            **stats,
            "ir_hotspot_mask":    hotspot_mask,
            "ir_analysis_status": status,
        })

    except Exception as exc:
        logger.error("IR damage analysis failed: %s", exc)
        result.update({
            "ir_healthy_avg_intensity": None,
            "ir_hotspot_avg_intensity": None,
            "delta_t_proxy":            None,
            "ir_damage_percent":        None,
            "ir_hotspot_mask":          None,
            "ir_analysis_status":       "error",
        })

    return result


# ──────────────────────────────────────────────────────────────────────────────
# SMOKE TEST  —  real classifier output + real IR image
# python damage_area_IR.py
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, ".")          # import classifier from same folder
    from classifier import classify_panel

    IR_PATH = r"C:\Users\riyasharma\Documents\Solar Project\final_IR\BackSheetDamage\BacksheetDamage(2).png"

    ir_array = cv2.imread(IR_PATH)
    if ir_array is None:
        print(f"[ERROR] Image not found: {IR_PATH}")
        sys.exit(1)

    # Step 1 — classifier
    clf_result = classify_panel(
        panel_id = 1,
        mode     = "ir",
        ir_array = ir_array,
        tmin     = 20.0,
        tmax     = 100.0,
    )

    # Step 2 — IR damage area  (pass same ir_array separately)
    result = analyze_ir_damage(clf_result, ir_array=ir_array)

    # print (skip mask array)
    printable = {k: v for k, v in result.items() if k != "ir_hotspot_mask"}
    print(json.dumps(printable, indent=2, default=str))

    print("\n========== IR DAMAGE RESULTS ==========")
    print(f"Class          : {result['main_category']} → {result['subcategory']}")
    print(f"Confidence     : {result['confidence']}")
    print(f"Healthy Avg    : {result['ir_healthy_avg_intensity']}")
    print(f"Hotspot Avg    : {result['ir_hotspot_avg_intensity']}")
    print(f"Delta T Proxy  : {result['delta_t_proxy']}")
    print(f"Damage Area %  : {result['ir_damage_percent']}%")
    print(f"Status         : {result['ir_analysis_status']}")
    print("=======================================")