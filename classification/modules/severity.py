"""
severity.py
============
Pipeline Position : Step 5 — FINAL STEP (after electrical_estimator.py)

Input  : clf_result — combined output of estimate_electrical()
         (must have: subcategory, ir_damage_percent, delta_t_corrected,
                     env_status, ff, delta_p, curve_type)

Output : clf_result extended with final severity fields

Severity Levels (doc):
    Mild     : delta_t < 70  AND area <= 10%
    Moderate : delta_t 70-95 OR  area 11-30%
    Severe   : delta_t >= 95 OR  area > 30%
    Critical : FullPanelUnresponsive / UnusualCooling / isc ~ 0

Escalation:
    If FF or delta_P indicate worse → escalate
    Curve hierarchy: FlatLine > Truncated > Distorted > Flattened > Normal

Usage
-----
>>> result = estimate_electrical(clf, pre_severity="Moderate")
>>> result = assess_severity(result)

New keys added:
{
    "severity"           : "Moderate",
    "recommended_action" : "Schedule maintenance within 30 days",
    "health_score"       : 62.4,
    "ir_valid"           : True,
    "escalated"          : False,
    "severity_reason"    : "delta_t=82.3 area=24.04%"
}
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

CRITICAL_SUBCATEGORIES = {
    "Full Panel Unresponsive",
    "Unusual Cooling / Full Panel Down",
}

ACTIONS = {
    "Mild":     "Monitor — schedule inspection within 90 days",
    "Moderate": "Schedule maintenance within 30 days",
    "Severe":   "Urgent maintenance required within 7 days",
    "Critical": "Immediate shutdown and replacement required",
}

CURVE_SEVERITY = {
    "Normal":    "Mild",
    "Flattened": "Moderate",
    "Distorted": "Severe",
    "Truncated": "Severe",
    "FlatLine":  "Critical",
    "Flat line": "Critical",
}

SEVERITY_ORDER = ["Mild", "Moderate", "Severe", "Critical"]


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def assess_severity(clf_result: dict) -> dict:
    """
    Final severity assessment with escalation from FF, delta_P, curve_type.

    Parameters
    ----------
    clf_result : dict — output of estimate_electrical()
                        must have:
                          subcategory, ir_damage_percent,
                          delta_t_corrected, env_status,
                          ff, delta_p, curve_type

    Returns
    -------
    dict — all clf_result keys + severity keys
    """
    result = dict(clf_result)

    subcategory   = clf_result.get("subcategory", "")
    ir_area       = clf_result.get("ir_damage_percent") or 0.0
    rgb_area      = clf_result.get("rgb_damage_percent") or 0.0
    damage_area   = max(ir_area, rgb_area)
    delta_t_corr  = clf_result.get("delta_t_corrected")
    env_status    = clf_result.get("env_status", "SKIPPED")
    ff            = clf_result.get("ff")
    delta_p       = clf_result.get("delta_p")
    curve_type    = clf_result.get("curve_type")
    isc           = clf_result.get("isc")

    # ── Critical class override ───────────────────────────────────────────────
    if subcategory in CRITICAL_SUBCATEGORIES or (isc is not None and isc == 0.0):
        result.update(_build_result(
            severity  = "Critical",
            reason    = f"subcategory='{subcategory}' is always Critical",
            escalated = False,
            delta_t   = delta_t_corr,
            area      = damage_area,
        ))
        return result

    # ── IR valid? ─────────────────────────────────────────────────────────────
    ir_valid = (env_status in ("ACCEPTED", "OPTIMIZED")) and (delta_t_corr is not None)

    # ── Base severity ─────────────────────────────────────────────────────────
    if ir_valid:
        base_severity, reason = _combined_severity(delta_t_corr, damage_area)
    else:
        base_severity, reason = _area_only_severity(damage_area)
        reason += " [IR invalid/skipped — area only]"

    # ── Escalation ────────────────────────────────────────────────────────────
    final_severity = base_severity
    escalated      = False

    ff_sev = _ff_severity(ff)
    if ff_sev and _is_higher(ff_sev, final_severity):
        final_severity = ff_sev
        reason        += f" | escalated by FF={ff:.3f}"
        escalated      = True

    dp_sev = _dp_severity(delta_p)
    if dp_sev and _is_higher(dp_sev, final_severity):
        final_severity = dp_sev
        reason        += f" | escalated by delta_P={delta_p:.1f}%"
        escalated      = True

    ct_sev = CURVE_SEVERITY.get(curve_type) if curve_type else None
    if ct_sev and _is_higher(ct_sev, final_severity):
        final_severity = ct_sev
        reason        += f" | escalated by curve={curve_type}"
        escalated      = True

    # ── Health score ──────────────────────────────────────────────────────────
    # 100 - damage_area - (delta_t_corrected × 0.5)
    dt_for_score  = delta_t_corr if delta_t_corr is not None else 0.0
    health_score  = max(0.0, round(100 - damage_area - (dt_for_score * 0.5), 2))

    logger.info(
        "Severity: %s | escalated=%s | health=%.1f | %s",
        final_severity, escalated, health_score, reason
    )

    result.update(_build_result(
        severity      = final_severity,
        reason        = reason,
        escalated     = escalated,
        delta_t       = delta_t_corr,
        area          = damage_area,
        ir_valid      = ir_valid,
        health_score  = health_score,
    ))

    return result


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _build_result(
    severity:     str,
    reason:       str,
    escalated:    bool,
    delta_t:      Optional[float],
    area:         float,
    ir_valid:     bool  = True,
    health_score: float = 0.0,
) -> dict:
    return {
        "severity":            severity,
        "recommended_action":  ACTIONS[severity],
        "health_score":        health_score,
        "ir_valid":            ir_valid,
        "escalated":           escalated,
        "severity_reason":     reason,
    }


def _combined_severity(delta_t: float, area: float):
    if delta_t >= 95 or area > 30:
        return "Severe",   f"delta_t={delta_t:.1f} area={area:.1f}%"
    if (70 <= delta_t < 95) or (11 <= area <= 30):
        return "Moderate", f"delta_t={delta_t:.1f} area={area:.1f}%"
    return "Mild",         f"delta_t={delta_t:.1f} area={area:.1f}%"


def _area_only_severity(area: float):
    if area > 30:
        return "Severe",   f"area={area:.1f}% > 30%"
    if area >= 11:
        return "Moderate", f"area={area:.1f}% 11-30%"
    return "Mild",         f"area={area:.1f}% <= 10%"


def _ff_severity(ff: Optional[float]) -> Optional[str]:
    if ff is None:
        return None
    if ff < 0.45: return "Critical"
    if ff < 0.55: return "Severe"
    if ff < 0.70: return "Moderate"
    return "Mild"


def _dp_severity(dp: Optional[float]) -> Optional[str]:
    if dp is None:
        return None
    if dp > 40:  return "Critical"
    if dp > 20:  return "Severe"
    if dp >= 5:  return "Moderate"
    return "Mild"


def _is_higher(new: str, current: str) -> bool:
    return SEVERITY_ORDER.index(new) > SEVERITY_ORDER.index(current)


# ──────────────────────────────────────────────────────────────────────────────
# SMOKE TEST  (python severity.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys, cv2
    sys.path.insert(0, ".")

    from modules.classifier               import classify_panel
    from modules.damage_area_IR           import analyze_ir_damage
    from modules.environmental_validation import validate_environment
    from modules.electrical_estimator     import estimate_electrical

    IR_PATH = r"C:\Users\riyasharma\Documents\Solar Project\final_IR\BackSheetDamage\BacksheetDamage(2).png"

    ir_array = cv2.imread(IR_PATH)
    if ir_array is None:
        print(f"[ERROR] Image not found: {IR_PATH}")
        sys.exit(1)

    # Step 1
    clf = classify_panel(1, "ir", ir_array=ir_array, tmin=20.0, tmax=100.0)

    # Step 2
    clf = analyze_ir_damage(clf, ir_array=ir_array)

    # Step 3
    clf = validate_environment(clf, irradiance=900, wind_speed=4.5)

    # Step 4
    clf = estimate_electrical(clf, pre_severity="Moderate")

    # Step 5 — FINAL
    result = assess_severity(clf)

    # print (skip mask)
    printable = {k: v for k, v in result.items() if k != "ir_hotspot_mask"}
    print(json.dumps(printable, indent=2, default=str))

    print("\n" + "=" * 60)
    print("SOLAR PANEL HEALTH REPORT")
    print("=" * 60)
    print(f"Class            : {result['main_category']} -> {result['subcategory']}")
    print(f"Confidence       : {result['confidence']}")
    print(f"IR Damage Area   : {result['ir_damage_percent']}%")
    print(f"Delta T Proxy    : {result['delta_t_proxy']}")
    print(f"Delta T Corrected: {result['delta_t_corrected']}")
    print(f"Env Status       : {result['env_status']}")
    print(f"Curve Type       : {result['curve_type']}")
    print(f"Fill Factor      : {result['ff']}")
    print(f"Power Loss       : {result['delta_p']}%")
    print(f"Severity         : {result['severity']}")
    print(f"Escalated        : {result['escalated']}")
    print(f"Health Score     : {result['health_score']}")
    print(f"Action           : {result['recommended_action']}")
    print(f"Reason           : {result['severity_reason']}")
    print("=" * 60)