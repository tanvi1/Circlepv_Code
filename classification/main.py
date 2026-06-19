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

from modules.classifier               import classify_panel
from modules.damage_area_IR           import analyze_ir_damage
from modules.RGB_Damage_area          import analyze_rgb_damage
from modules.environmental_validation import validate_environment
from modules.electrical_estimator     import estimate_electrical
from modules.severity                 import assess_severity

BACKSHEET_FLAG = 1             # 1 for v2 (Clean), 2 for Legacy Backsheet
CLIENT_MODE    = True          # True = Use Client scripts, False = Use Default scripts

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  —  edit these before running
# ──────────────────────────────────────────────────────────────────────────────

# IR_PATH    = r"imgs\hotspot(3).png"
IR_PATH = None
# RGB_PATH   = r"imgs\Image.jpg"          # set path if RGB image available, else None
RGB_PATH   = r"imgs\SnailTrails_Microcracks1.png"          # set path if RGB image available, else None
# RGB_PATH = None

PANEL_ID   = 1
MODE       = "rgb"          # "rgb" | "ir" | "both"

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

from modules.classifier import LABEL_MAP

def _extract_all_defects(clf):
    """Extracts all classes > threshold and calculates W_RGB / W_IR per defect."""
    targets = {}

    def _add(norm_label, conf, modality):
        if conf < 0.25 or norm_label == "normal": return
        cat, sub, _ = LABEL_MAP.get(norm_label, ("Unknown", norm_label, modality))
        
        if sub not in targets:
            targets[sub] = {
                "main_category": cat, "subcategory": sub, 
                "conf_rgb": 0.0, "conf_ir": 0.0
            }
            
        if modality == "RGB":
            targets[sub]["conf_rgb"] = max(targets[sub]["conf_rgb"], conf)
        else:
            targets[sub]["conf_ir"] = max(targets[sub]["conf_ir"], conf)

    rgb_p = clf.get("rgb_prediction") or {}
    if "all_probs" in rgb_p:
         for k, v in rgb_p["all_probs"].items(): _add(k.lower().replace(" ","").replace("_","").replace("-",""), v, "RGB")
    elif "all_detections" in rgb_p:
         for d in rgb_p["all_detections"]: _add(d["label"], d["confidence"], "RGB")

    ir_p = clf.get("ir_prediction") or {}
    if "all_probs" in ir_p:
         for k, v in ir_p["all_probs"].items(): _add(k.lower().replace(" ","").replace("_","").replace("-",""), v, "IR")
    elif "all_detections" in ir_p:
         for d in ir_p["all_detections"]: _add(d["label"], d["confidence"], "IR")

    if not targets:
         return [{"subcategory": "No Defect", "main_category": "Healthy", "confidence": 1.0, "primary": "RGB", "conf_rgb": 1.0, "conf_ir": 0.0, "w_rgb": 1.0, "w_ir": 0.0}]

    for t in targets.values():
        total = t["conf_rgb"] + t["conf_ir"]
        t["w_rgb"] = round(t["conf_rgb"] / total, 2) if total > 0 else 0.0
        t["w_ir"]  = round(t["conf_ir"] / total, 2) if total > 0 else 0.0
        t["confidence"] = max(t["conf_rgb"], t["conf_ir"])
        t["primary"] = "RGB" if t["conf_rgb"] >= t["conf_ir"] else "IR"

    return list(targets.values())

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
    backsheet_flag: int = BACKSHEET_FLAG,
    client_mode: bool = CLIENT_MODE,
) -> dict:

    ir_array  = cv2.imread(ir_path)  if ir_path  else None
    rgb_array = cv2.imread(rgb_path) if rgb_path else None

    if ir_array is None and rgb_array is None:
        raise FileNotFoundError("No valid image found. Check IR_PATH / RGB_PATH.")

    print("Step 1 — Classification...")
    clf_base = classify_panel(panel_id, mode, rgb_array, ir_array, tmin, tmax)

    defects_to_process = _extract_all_defects(clf_base)
    all_reports = []

    overall_dp = 0.0
    worst_severity = "Mild"
    severity_ranks = {"Mild": 1, "Moderate": 2, "Severe": 3, "Critical": 4}

    print(f"Found {len(defects_to_process)} target defect(s) on panel.")

    for i, d in enumerate(defects_to_process, 1):
        print(f"\n--- Analyzing Defect {i}: {d['subcategory']} ---")
        
        # Localize standard dictionary for this loop
        local_clf = dict(clf_base)
        local_clf.update({
            "main_category": d["main_category"],
            "subcategory": d["subcategory"],
            "confidence": d["confidence"],
            "primary": d["primary"],
            "conf_rgb": d["conf_rgb"],
            "conf_ir": d["conf_ir"],
            "w_rgb": d["w_rgb"],
            "w_ir": d["w_ir"]
        })

        # IR Area
        if ir_array is not None:
            local_clf = analyze_ir_damage(local_clf, ir_array=ir_array)
        else:
            local_clf.update({
                "delta_t_proxy": None,
                "ir_damage_percent": None,
                "ir_analysis_status": "skipped",
            })

        # RGB Area
        if rgb_array is not None:
            # Client Mapping for RGB processing routing
            lbl_map = {"Hot Spots – Soiling/Shading (Hard, Bird Poop)": "HardShading", "Soft Shading / Soiling": "SoftShading_Soiling", 
                       "Glass Break": "GlassBreak", "Snail Trail / Microcrack": "SnailTrails_Microcracks"}
            safe_label = lbl_map.get(d["subcategory"], d["subcategory"])
            
            try:
                rgb_res = analyze_rgb_damage(rgb_array, safe_label, backsheet_flag, client_mode)
                local_clf["rgb_damage_percent"] = rgb_res.damage_percentage
            except Exception:
                local_clf["rgb_damage_percent"] = None
        else:
            local_clf["rgb_damage_percent"] = None

        # Env + IV + Severity
        local_clf = validate_environment(local_clf, irradiance, wind_speed)
        local_clf = estimate_electrical(local_clf, pre_severity="Moderate")
        local_result = assess_severity(local_clf)

        # Client Mapping for Final Output Naming
        client_mapping = {
            "Snail Trail / Microcrack": "Excessive Heating",
            "Hot Spots": "Critical Overheating",
            "Full Panel Unresponsive": "Excessive Heating",
            "Hot Spots – Soiling/Shading (Hard, Bird Poop)": "Critical Overheating",
            "HardShading": "Critical Overheating",
            "Hot Spots – Soiling/Shading (Soft, Atmos.)": "Critical Overheating",
            "Soft Shading / Soiling": "Critical Overheating",
        }
        local_result["main_category"] = client_mapping.get(local_result["subcategory"], local_result["main_category"])

        # Aggregate Metrics
        dp = local_result.get("delta_p") or 0.0
        overall_dp += dp

        sev = local_result.get("severity", "Mild")
        if severity_ranks.get(sev, 1) > severity_ranks.get(worst_severity, 1):
            worst_severity = sev

        all_reports.append(local_result)

    # Apply Client Sorting Logic: Confidence -> Severity -> Curve Hierarchy
    curve_ranks = {"FlatLine": 5, "Flat line": 5, "Truncated": 4, "Distorted": 3, "Flattened": 2, "Normal": 1}
    all_reports.sort(
        key=lambda r: (
            round(r.get("confidence", 0.0) / 0.1), # Group confidences within ~10%
            severity_ranks.get(r.get("severity", "Mild"), 1),
            curve_ranks.get(r.get("curve_type", "Normal"), 1),
            r.get("confidence", 0.0) # Exact tie breaker
        ),
        reverse=True
    )

    rank_labels = ["Primary", "Secondary", "Tertiary"]
    for i, r in enumerate(all_reports):
        r["defect_rank"] = rank_labels[i] if i < len(rank_labels) else f"Rank-{i+1}"

    overall_health = max(0.0, round(100.0 - (overall_dp * 1.5), 2))

    final_output = {
        "panel_id": panel_id,
        "overall_severity": worst_severity,
        "overall_health_score": overall_health,
        "total_power_loss_pct": min(round(overall_dp, 2), 100.0),
        "reliability": _reliability_tag(mode, rgb_array is not None, ir_array is not None),
        "detected_defects": []
    }

    for r in all_reports:
        final_output["detected_defects"].append({
            "defect_rank": r["defect_rank"],
            "defect_type": f"{r['main_category']} -> {r['subcategory']}",
            "confidence": r["confidence"],
            "primary_modality": r.get("primary", "Unknown"),
            "conf_rgb": r.get("conf_rgb", 0.0),
            "conf_ir": r.get("conf_ir", 0.0),
            "w_rgb": r.get("w_rgb", 0.0),
            "w_ir": r.get("w_ir", 0.0),
            "ir_damage_percent": r.get("ir_damage_percent"),
            "rgb_damage_percent": r.get("rgb_damage_percent"),
            "delta_t_proxy": r.get("delta_t_proxy"),
            "delta_t_corrected": r.get("delta_t_corrected"),
            "env_status": r.get("env_status"),
            "curve_type": r.get("curve_type"),
            "fill_factor": r.get("ff"),
            "power_loss": r.get("delta_p"),
            "voc": r.get("voc"),
            "isc": r.get("isc"),
            "vmp": r.get("vmp_adjusted"),
            "imp": r.get("imp_adjusted"),
            "severity": r.get("severity"),
            "severity_reason": r.get("severity_reason"),
            "escalated": r.get("escalated", False)
        })

    return final_output


# ──────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER  (matches doc Example Report Output)
# ──────────────────────────────────────────────────────────────────────────────

def print_report(result: dict):
    print("\n" + "=" * 60)
    print("      SOLAR PANEL HEALTH REPORT (MULTI-DEFECT)")
    print("=" * 60)
    print(f"Panel ID             : {result['panel_id']}")
    print(f"Overall Severity     : {result['overall_severity']}")
    print(f"Overall Health Score : {result['overall_health_score']}%")
    print(f"Total Power Loss     : {result['total_power_loss_pct']}%")
    print(f"Reliability          : {result['reliability']}")
    print("-" * 60)
    print(f"DETECTED DEFECTS ({len(result['detected_defects'])} found):")

    for i, d in enumerate(result['detected_defects'], 1):
        print(f"  [{i}] {d['defect_rank']} Defect : {d['defect_type']}")
        print(f"      Confidence : RGB {d.get('conf_rgb')}  IR {d.get('conf_ir')}  ->  W_RGB={d.get('w_rgb')}  W_IR={d.get('w_ir')}")
        if d['ir_damage_percent'] is not None:
            print(f"      IR Area    : {d['ir_damage_percent']}%")
        if d['rgb_damage_percent'] is not None:
            print(f"      RGB Area   : {d['rgb_damage_percent']}%")
        print(f"      Delta T    : proxy={d.get('delta_t_proxy')} | corr={d.get('delta_t_corrected')}")
        print(f"      Env Status : {d.get('env_status')}")
        print(f"      IV Curve   : {d.get('curve_type')} | FF: {d.get('fill_factor')}")
        print(f"      Voc / Isc  : {d.get('voc')}V / {d.get('isc')}A")
        print(f"      Vmp / Imp  : {d.get('vmp')}V / {d.get('imp')}A")
        print(f"      Power Loss : {d.get('power_loss')}%")
        print(f"      Severity   : {d.get('severity')}")
        print(f"      Reason     : {d.get('severity_reason')}")
        print(f"      Escalated  : {d.get('escalated')}")
        print("")

    # # Fulfilling the client's request for single-defect panels
    # if len(result['detected_defects']) == 1:
    #     print("  [2] Secondary Defect: None")
    #     print("")
        
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