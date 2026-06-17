"""
main.py
========
Solar Panel Damage Detection Pipeline — IR Mode

Pipeline order (doc):
    Step 1 : classify_panel()         — YOLO classification, confidence, modality fusion
    Step 2 : analyze_ir_damage()      — IR hotspot mask, damage area %, delta_T proxy
    Step 3 : validate_environment()   — HKO correction, IV curve simulation
    Step 4 : estimate_electrical()    — FF, delta_P, Vmp/Imp adjusted
    Step 5 : assess_severity()        — final severity, health score, recommended action

Final output matches doc Example Report Output:
    - Main Category → Subcategory
    - Confidence (weighted W_RGB, W_IR)
    - Damage Area % (IR)
    - Severity (Mild / Moderate / Severe / Critical)
    - IV Curve Type, FF, delta_P
    - Reliability Tag
    - Recommended Action

Usage
-----
    python main.py
    (edit IR_PATH, TMIN, TMAX, IRRADIANCE, WIND_SPEED below)
"""

import sys
import json
import cv2

sys.path.insert(0, ".")

from classifier               import classify_panel
from damage_area_IR           import analyze_ir_damage
from RGB_Damage_area          import analyze_rgb_damage
from environmental_validation import validate_environment
from electrical_estimator     import estimate_electrical
from severity                 import assess_severity


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  —  edit these before running
# ──────────────────────────────────────────────────────────────────────────────

IR_PATH    = r"C:\Users\riyasharma\Documents\Solar Project\classification\imgs\hotspot(3).png"
RGB_PATH   = None          # set path if RGB image available, else None

PANEL_ID   = 1
MODE       = "ir"          # "rgb" | "ir" | "both"

TMIN       = 20.0          # IR scale bar min temp (°C)
TMAX       = 100.0         # IR scale bar max temp (°C)

IRRADIANCE = 900           # W/m²  (from drone/site)
WIND_SPEED = 4.5           # m/s   (from drone/site)


# ──────────────────────────────────────────────────────────────────────────────
# RELIABILITY TAG  (doc rules)
# ──────────────────────────────────────────────────────────────────────────────

def _reliability_tag(mode: str, rgb_ok: bool, ir_ok: bool) -> str:
    if rgb_ok and ir_ok:
        return "RGB + IR available — Combined weighting"
    if rgb_ok and not ir_ok:
        return "RGB available — RGB primary"
    if ir_ok and not rgb_ok:
        return "IR available — IR primary"
    return "No modality available"


# ──────────────────────────────────────────────────────────────────────────────
# WEIGHTED CONFIDENCE  (doc formula)
# W_RGB = conf_rgb / (conf_rgb + conf_ir)
# W_IR  = conf_ir  / (conf_rgb + conf_ir)
# ──────────────────────────────────────────────────────────────────────────────

def _weighted_confidence(rgb_pred, ir_pred):
    conf_rgb = float((rgb_pred or {}).get("confidence", 0.0))
    conf_ir  = float((ir_pred  or {}).get("confidence", 0.0))
    total    = conf_rgb + conf_ir
    if total == 0:
        return None, None
    return round(conf_rgb / total, 2), round(conf_ir / total, 2)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    ir_path:    str   = IR_PATH,
    rgb_path:   str   = RGB_PATH,
    panel_id:   int   = PANEL_ID,
    mode:       str   = MODE,
    tmin:       float = TMIN,
    tmax:       float = TMAX,
    irradiance: float = IRRADIANCE,
    wind_speed: float = WIND_SPEED,
) -> dict:

    # ── load arrays ───────────────────────────────────────────────────────────
    ir_array  = cv2.imread(ir_path)  if ir_path  else None
    rgb_array = cv2.imread(rgb_path) if rgb_path else None

    if ir_array is None and rgb_array is None:
        raise FileNotFoundError("No valid image found. Check IR_PATH / RGB_PATH.")

    # ── Step 1: classifier ────────────────────────────────────────────────────
    print("Step 1 — Classification...")
    clf = classify_panel(
        panel_id  = panel_id,
        mode      = mode,
        ir_array  = ir_array,
        rgb_array = rgb_array,
        tmin      = tmin,
        tmax      = tmax,
    )

    # ── Step 2: IR damage area ────────────────────────────────────────────────
    if ir_array is not None:
        print("Step 2 — IR Damage Area...")
        clf = analyze_ir_damage(clf, ir_array=ir_array)
    else:
        clf.update({
            "ir_healthy_avg_intensity": None,
            "ir_hotspot_avg_intensity": None,
            "delta_t_proxy":            None,
            "ir_damage_percent":        None,
            "ir_hotspot_mask":          None,
            "ir_analysis_status":       "skipped",
        })

    # ── Step 3: environmental validation ─────────────────────────────────────
    print("Step 3 — Environmental Correction + IV Curve...")
    clf = validate_environment(clf, irradiance=irradiance, wind_speed=wind_speed)

    # ── Step 4: electrical estimate ───────────────────────────────────────────
    print("Step 4 — Electrical Estimation...")
    clf = estimate_electrical(clf, pre_severity="Moderate")

    # ── Step 5: severity ──────────────────────────────────────────────────────
    print("Step 5 — Severity Assessment...")
    result = assess_severity(clf)

    # ── build final output (doc fields only) ─────────────────────────────────
    rgb_pred = result.get("rgb_prediction")
    ir_pred  = result.get("ir_prediction")
    w_rgb, w_ir = _weighted_confidence(rgb_pred, ir_pred)

    conf_rgb = float((rgb_pred or {}).get("confidence", 0.0)) or None
    conf_ir  = float((ir_pred  or {}).get("confidence", 0.0)) or None

    reliability = _reliability_tag(
        mode    = mode,
        rgb_ok  = rgb_array is not None,
        ir_ok   = ir_array  is not None,
    )

    final = {
        # ── classification ────────────────────────────────────────────────────
        "panel_id":           panel_id,
        "main_category":      result["main_category"],
        "subcategory":        result["subcategory"],

        # ── confidence (doc: W_RGB, W_IR) ────────────────────────────────────
        "confidence_rgb":     conf_rgb,
        "confidence_ir":      conf_ir,
        "w_rgb":              w_rgb,
        "w_ir":               w_ir,

        # ── damage area ───────────────────────────────────────────────────────
        "ir_damage_percent":  result.get("ir_damage_percent"),
        "rgb_damage_percent": None,          # populated when damage_area_RGB.py added

        # ── thermal ───────────────────────────────────────────────────────────
        "delta_t_proxy":      result.get("delta_t_proxy"),
        "delta_t_corrected":  result.get("delta_t_corrected"),
        "env_status":         result.get("env_status"),

        # ── IV curve ─────────────────────────────────────────────────────────
        "curve_type":         result.get("curve_type"),
        "fill_factor":        result.get("ff"),
        "delta_p":            result.get("delta_p"),
        "voc":                result.get("voc"),
        "isc":                result.get("isc"),
        "vmp":                result.get("vmp_adjusted"),
        "imp":                result.get("imp_adjusted"),

        # ── severity (doc levels) ─────────────────────────────────────────────
        "severity":           result["severity"],
        "severity_reason":    result["severity_reason"],
        "escalated":          result["escalated"],

        # ── health + action ───────────────────────────────────────────────────
        "health_score":       result["health_score"],
        # "recommended_action": result["recommended_action"],

        # ── reliability tag (doc) ─────────────────────────────────────────────
        "reliability":        reliability,
    }

    return final


# ──────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER  (matches doc Example Report Output)
# ──────────────────────────────────────────────────────────────────────────────

def print_report(result: dict):
    print("\n" + "=" * 60)
    print("      SOLAR PANEL HEALTH REPORT")
    print("=" * 60)

    print(f"Panel ID             : {result['panel_id']}")
    print(f"Defect Detected      : {result['main_category']} → {result['subcategory']}")

    # confidence
    if result["w_rgb"] is not None and result["w_ir"] is not None:
        print(f"Confidence           : RGB {result['confidence_rgb']}  IR {result['confidence_ir']}"
              f"  →  W_RGB={result['w_rgb']}  W_IR={result['w_ir']}")
    elif result["confidence_ir"] is not None:
        print(f"Confidence           : IR {result['confidence_ir']}")
    elif result["confidence_rgb"] is not None:
        print(f"Confidence           : RGB {result['confidence_rgb']}")

    # damage area
    if result["ir_damage_percent"] is not None:
        print(f"IR Damage Area %     : {result['ir_damage_percent']}%")
    if result["rgb_damage_percent"] is not None:
        print(f"RGB Damage Area %    : {result['rgb_damage_percent']}%")

    # thermal
    print(f"Delta T (proxy)      : {result['delta_t_proxy']}")
    print(f"Delta T (corrected)  : {result['delta_t_corrected']}")
    print(f"Env Correction       : {result['env_status']}")

    # IV curve
    print(f"IV Curve Type        : {result['curve_type']}")
    print(f"Fill Factor (FF)     : {result['fill_factor']}")
    print(f"Power Loss (dP)      : {result['delta_p']}%")
    print(f"Voc / Isc            : {result['voc']}V / {result['isc']}A")
    print(f"Vmp / Imp            : {result['vmp']}V / {result['imp']}A")

    # severity
    print(f"Severity             : {result['severity']}")
    print(f"Severity Reason      : {result['severity_reason']}")
    print(f"Escalated            : {result['escalated']}")

    # health
    print(f"Health Score         : {result['health_score']}%")
    # print(f"Recommended Action   : {result['recommended_action']}")
    print(f"Reliability          : {result['reliability']}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_pipeline()
    print_report(result)

    # also dump full JSON (skip mask)
    print("\n--- Full JSON Output ---")
    print(json.dumps(result, indent=2, default=str))