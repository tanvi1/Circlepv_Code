"""
classifier.py
=============
Solar Panel Damage Classifier  —  Step 3 of pipeline
 
Responsibility (ONLY):
  1. Load RGB + IR YOLO models
  2. Run inference on panel crop(s)
  3. Fuse predictions (confidence-weighted) when both modalities available
  4. Return main_category, subcategory, confidence + raw model outputs
 
Does NOT do:
  - IR temperature mapping       → damage_area.py
  - Damage area %                → damage_area.py
  - Severity levels              → severity_fusion.py
  - Reliability tags             → severity_fusion.py
  - Environmental corrections    → env_correction.py
  - IV curve logic               → iv_curve.py
 
Usage
-----
>>> from classifier import classify_panel
>>> result = classify_panel(
...     panel_id  = 1,
...     mode      = "both",      # "rgb" | "ir" | "both"
...     rgb_array = rgb_crop,    # np.ndarray H×W×3, panel crop
...     ir_array  = ir_crop,     # np.ndarray H×W or H×W×3, panel crop
...     tmin      = 20.0,        # only needed for mode="ir"/"both"
...     tmax      = 100.0,
... )
 
Output dict
-----------
{
    "panel_id":       1,
    "main_category":  "Excessive Heating",
    "subcategory":    "Snail Trail / Microcrack",
    "confidence":     0.87,
    "primary":        "RGB",          # which modality drove the decision
    "tmin":           20.0,           # None if mode="rgb"
    "tmax":           100.0,          # None if mode="rgb"
    "rgb_prediction": {...},          # raw YOLO result dict or None
    "ir_prediction":  {...},          # raw YOLO result dict or None
}
"""
 
from __future__ import annotations
 
import logging
from typing import Optional
 
import numpy as np
 
logger = logging.getLogger(__name__)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 1.  MODEL PATHS  —  edit before running
# ──────────────────────────────────────────────────────────────────────────────
 
MODEL_PATHS = {
    "rgb": "models/rgb_yolo.pt",
    "ir":  "models/ir_yolo.pt",
}
 
_RGB_MODEL = None
_IR_MODEL  = None
 
 
def _load_models() -> None:
    """Lazy-load YOLO models once on first call."""
    global _RGB_MODEL, _IR_MODEL
    try:
        from ultralytics import YOLO
        if _RGB_MODEL is None:
            _RGB_MODEL = YOLO(MODEL_PATHS["rgb"])
            logger.info("RGB model loaded: %s", MODEL_PATHS["rgb"])
        if _IR_MODEL is None:
            _IR_MODEL = YOLO(MODEL_PATHS["ir"])
            logger.info("IR  model loaded: %s", MODEL_PATHS["ir"])
    except Exception as exc:
        logger.error("Model load failed: %s", exc)
        raise
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 2.  LABEL → CATEGORY MAP  (from doc taxonomy)
# ──────────────────────────────────────────────────────────────────────────────
# Format: yolo_label → (main_category, subcategory, preferred_primary_modality)
 
# ── Label normalizer ──────────────────────────────────────────────────────────
# Strips spaces, underscores, hyphens and lowercases → single canonical key.
def _norm(label: str) -> str:
    return label.lower().replace(" ", "").replace("_", "").replace("-", "")
 
 
# LABEL_MAP keys = _norm(YOLO class name)
#
# RGB model classes: BackSheetDamage, BypassDiode, Corrosion_Discoloration_Delamination,
#                    GlassBreak, HardShading_BirdPoop, HotSpots, InverterBatteryDamage,
#                    Normal, SnailTrails_Microcracks, SoftShading_Soiling
#
# IR  model classes: BackSheetDamage, BypassDiode, Corrosion_Discoloration_Delamination,
#                    Excessive_Cooling, GlassBreak, HardShading_BirdPoop, HotSpots,
#                    Normal, PID, SnailTrails_Microcracks, SoftShading_Soiling
#
# Format: norm_key -> (main_category, subcategory, preferred_primary_modality)
 
LABEL_MAP: dict[str, tuple[str, str, str]] = {
    # shared (both models)
    _norm("BackSheetDamage"):                      ("Back Sheet Damage",       "Backsheet Image",                               "Both"),
    _norm("BypassDiode"):                          ("Bypass Diode",            "Junction Box Fault",                            "IR"),
    _norm("Corrosion_Discoloration_Delamination"): ("Excessive Heating",       "EVA Delamination / Discoloration / Corrosion",  "Both"),
    _norm("GlassBreak"):                           ("Glass Break",             "Glass Break",                                   "RGB"),
    _norm("HardShading_BirdPoop"):                 ("Critical Overheating",    "Hot Spots – Soiling/Shading (Hard, Bird Poop)", "Both"),
    _norm("HotSpots"):                             ("Critical Overheating",    "Hot Spots",                                     "IR"),
    _norm("SnailTrails_Microcracks"):              ("Excessive Heating",       "Snail Trail / Microcrack",                      "Both"),
    _norm("SoftShading_Soiling"):                  ("Excessive Heating",       "Soft Shading / Soiling",                        "Both"),
    _norm("Normal"):                               ("Healthy",                 "No Defect",                                     "RGB"),
    # RGB-only
    _norm("InverterBatteryDamage"):                ("Inverter/Battery Damage", "Inverter / Battery Damage",                     "RGB"),
    # IR-only
    _norm("PID"):                                  ("Excessive Heating",       "PID",                                           "IR"),
    _norm("Excessive_Cooling"):                    ("Excessive Cooling",       "Unusual Cooling / Full Panel Down",              "IR"),
}
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 3.  YOLO INFERENCE
# ──────────────────────────────────────────────────────────────────────────────
 
def _run_yolo(model, image: np.ndarray) -> dict | None:
    """
    Run one YOLO model on an image crop.
    Handles both classification (probs) and detection (boxes) task heads.
 
    Returns
    -------
    {
        "label":      str,   # top predicted class, snake_case
        "confidence": float,
        # classification task only:
        "all_probs":  {class_name: prob, ...}
        # detection task only:
        "boxes":      [[x1,y1,x2,y2], ...]
    }
    or None on error / no model.
    """
    if model is None or image is None:
        return None
    try:
        results = model(image, verbose=False)
        r = results[0]
 
        # classification head
        if hasattr(r, "probs") and r.probs is not None:
            top_idx  = int(r.probs.top1)
            top_conf = float(r.probs.top1conf)
            label    = _norm(r.names[top_idx])
            return {
                "label":      label,
                "confidence": round(top_conf, 4),
                "all_probs":  {r.names[i]: round(float(p), 4)
                               for i, p in enumerate(r.probs.data.tolist())},
            }
 
        # detection head  — pick highest-confidence box
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes):
            confs  = r.boxes.conf.cpu().numpy()
            best   = int(np.argmax(confs))
            cls_id = int(r.boxes.cls[best].cpu().numpy())
            label  = _norm(r.names[cls_id])
            
            # Adding All Detections
            all_dets = []
            for i, c in enumerate(confs):
                if c > 0.25:
                    all_dets.append({
                        "label": _norm(r.names[int(r.boxes.cls[i].cpu().numpy())]),
                        "confidence": float(c)
                    })
                    
            return {
                "label":      label,
                "confidence": round(float(confs[best]), 4),
                "all_detections": all_dets,
                "boxes":      r.boxes.xyxy.cpu().numpy().tolist(),
            }
 
        # nothing detected → treat as healthy
        return {"label": _norm("Normal"), "confidence": 1.0, "boxes": []}
 
    except Exception as exc:
        logger.warning("YOLO inference error: %s", exc)
        return None
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 4.  CONFIDENCE-WEIGHTED FUSION  (doc formula)
# ──────────────────────────────────────────────────────────────────────────────
 
def _fuse(
    rgb_pred: dict | None,
    ir_pred:  dict | None,
) -> tuple[str, str, float, str]:
    """
    Fuse RGB + IR predictions using doc formula:
        W_RGB = conf_rgb / (conf_rgb + conf_ir)
        W_IR  = conf_ir  / (conf_rgb + conf_ir)
 
    Returns (main_category, subcategory, fused_confidence, primary_modality)
    """
    rgb_label = (rgb_pred or {}).get("label", "healthy")
    ir_label  = (ir_pred  or {}).get("label", "healthy")
    rgb_conf  = float((rgb_pred or {}).get("confidence", 0.0))
    ir_conf   = float((ir_pred  or {}).get("confidence", 0.0))
 
    total = rgb_conf + ir_conf
    if total == 0:
        cat, sub, pri = LABEL_MAP.get(_norm("Normal"), ("Healthy", "No Defect", "RGB"))
        return cat, sub, 0.0, pri
 
    w_rgb = rgb_conf / total
    w_ir  = ir_conf  / total
 
    # weighted score per label
    scores: dict[str, float] = {}
    if rgb_pred:
        scores[rgb_label] = scores.get(rgb_label, 0.0) + w_rgb * rgb_conf
    if ir_pred:
        scores[ir_label]  = scores.get(ir_label,  0.0) + w_ir  * ir_conf
 
    best_label = max(scores, key=scores.__getitem__)
    best_conf  = round(scores[best_label], 4)
 
    cat, sub, pri = LABEL_MAP.get(
        best_label,
        ("Unknown", best_label.replace("_", " ").title(), "RGB"),
    )
    return cat, sub, best_conf, pri
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 5.  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────
 
def classify_panel(
    panel_id:  int,
    mode:      str,
    rgb_array: Optional[np.ndarray] = None,
    ir_array:  Optional[np.ndarray] = None,
    tmin:      float = 20.0,
    tmax:      float = 100.0,
) -> dict:
    """
    Classify a single solar panel crop.
 
    Parameters
    ----------
    panel_id  : int   — unique panel ID
    mode      : str   — "rgb" | "ir" | "both"
    rgb_array : ndarray | None  — H×W×3 panel crop
    ir_array  : ndarray | None  — H×W or H×W×3 panel crop
    tmin      : float — IR scale bar min temp (°C), passed through for damage_area.py
    tmax      : float — IR scale bar max temp (°C), passed through for damage_area.py
 
    Returns
    -------
    dict — see module docstring for full schema
    """
    mode = mode.lower().strip()
    if mode not in ("rgb", "ir", "both"):
        raise ValueError(f"mode must be 'rgb', 'ir', or 'both'; got '{mode}'")
 
    _load_models()
 
    # ── inference ──
    rgb_pred: dict | None = None
    ir_pred:  dict | None = None
 
    if mode in ("rgb", "both") and rgb_array is not None:
        rgb_pred = _run_yolo(_RGB_MODEL, rgb_array)
 
    if mode in ("ir", "both") and ir_array is not None:
        ir_pred = _run_yolo(_IR_MODEL, ir_array)
 
    # ── category decision ──
    if mode == "rgb":
        label         = (rgb_pred or {}).get("label", "healthy")
        conf          = float((rgb_pred or {}).get("confidence", 0.0))
        cat, sub, pri = LABEL_MAP.get(label, ("Unknown", label, "RGB"))
        pri           = "RGB"
 
    elif mode == "ir":
        label         = (ir_pred or {}).get("label", "healthy")
        conf          = float((ir_pred or {}).get("confidence", 0.0))
        cat, sub, pri = LABEL_MAP.get(label, ("Unknown", label, "IR"))
        pri           = "IR"
 
    else:  # both
        cat, sub, conf, pri = _fuse(rgb_pred, ir_pred)
 
    return {
        "panel_id":       panel_id,
        "main_category":  cat,
        "subcategory":    sub,
        "confidence":     conf,
        "primary":        pri,
        # passed through so downstream steps don't need to re-carry them
        "tmin":           tmin if mode != "rgb" else None,
        "tmax":           tmax if mode != "rgb" else None,
        # raw outputs — consumed by damage_area.py
        "rgb_prediction": rgb_pred,
        "ir_prediction":  ir_pred,
        "ir_array":       ir_array if mode != "rgb" else None,
        "rgb_array":      rgb_array if mode != "ir" else None,
    }

# ──────────────────────────────────────────────────────────────────────────────
# 6.  SMOKE TEST  (python classifier.py)
# ──────────────────────────────────────────────────────────────────────────────
import cv2
if __name__ == "__main__":
    import json

    # test_rgb = cv2.imread("GlassBreak(13).png")
    test_ir  = cv2.imread("hotspot(3).png")

    for test_mode, kw in [
        # ("rgb",  {"rgb_array": test_rgb}),
        ("ir",   {"ir_array": test_ir, "tmin": 20.0, "tmax": 100.0}),
        # ("both", {"rgb_array": test_rgb, "ir_array": test_ir, "tmin": 20.0, "tmax": 100.0}),
    ]:
        print(f"\n=== mode={test_mode} ===")
        try:
            result = classify_panel(panel_id=1, mode=test_mode, **kw)
            print(json.dumps(result, indent=2, default=str))
        except Exception as e:
            print(f"[SKIP – models not loaded] {e}")