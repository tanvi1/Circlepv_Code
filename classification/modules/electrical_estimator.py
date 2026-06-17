"""
electrical_estimator.py
========================
Pipeline Position : Step 4 — after environmental_validation.py

Input  : clf_result — combined output of validate_environment()
         (must have: subcategory, env_status, fill_factor, power_loss, curve_type)

Output : clf_result extended with electrical estimate fields

Does NOT do:
    - Classification        → classifier.py
    - IR damage area        → damage_area_IR.py
    - Env correction        → environmental_validation.py
    - Final severity        → severity.py

NOTE: All values ESTIMATED from lookup table.
      When real IV data available, replace estimate_electrical() with real extractor.

Usage
-----
>>> result = validate_environment(clf, irradiance=900, wind_speed=4.5)
>>> result = estimate_electrical(result)

New keys added:
{
    "ff"        : 0.612,        # Fill Factor (from IV simulator, severity-adjusted)
    "delta_p"   : 19.58,        # Power loss %
    "voc"       : 38.5,
    "isc"       : 6.5,
    "vmp"       : 26.84,        # severity-adjusted
    "imp"       : 5.94,         # severity-adjusted
    "electrical_note" : "ESTIMATED — replace with real IV data when available"
}
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# SEVERITY FACTOR — reduces Vmp/Imp based on severity
# (applied AFTER env_validation already gives base IV values)
# ──────────────────────────────────────────────────────────────────────────────

SEVERITY_FACTOR = {
    "Mild":     0.97,
    "Moderate": 0.88,
    "Severe":   0.75,
    "Critical": 0.50,
}

HEALTHY_VMP   = 32.0
HEALTHY_IMP   = 8.0
HEALTHY_POWER = HEALTHY_VMP * HEALTHY_IMP   # 256.0 W


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def estimate_electrical(
    clf_result:       dict,
    pre_severity:     str = "Moderate",   # rough severity before severity.py runs
) -> dict:
    """
    Estimate electrical parameters using IV values from environmental_validation
    and apply severity-based degradation factor.

    Parameters
    ----------
    clf_result   : dict — output of validate_environment()
                          must have: voc, isc, vmp, imp, fill_factor, power_loss,
                                     curve_type, env_status
    pre_severity : str  — rough severity estimate used ONLY to adjust Vmp/Imp
                          ("Mild" | "Moderate" | "Severe" | "Critical")
                          Will be overridden by severity.py in next step.

    Returns
    -------
    dict — all clf_result keys + electrical estimate keys
    """
    result = dict(clf_result)

    # ── base IV values from env_validation (iv_curve_simulator) ──────────────
    voc = clf_result.get("voc")
    isc = clf_result.get("isc")
    vmp = clf_result.get("vmp")
    imp = clf_result.get("imp")

    if None in (voc, isc, vmp, imp):
        logger.warning("IV values missing — electrical estimate skipped")
        result.update({
            "ff":               None,
            "delta_p":          None,
            "vmp_adjusted":     None,
            "imp_adjusted":     None,
            "electrical_note":  "SKIPPED — IV values not available",
        })
        return result

    # ── apply severity factor to Vmp / Imp ───────────────────────────────────
    factor      = SEVERITY_FACTOR.get(pre_severity, 0.88)
    vmp_adj     = round(vmp * factor, 2)
    imp_adj     = round(imp * factor, 2)

    # ── FF = (Vmp_adj × Imp_adj) / (Voc × Isc) ───────────────────────────────
    if voc > 0 and isc > 0:
        ff = round((vmp_adj * imp_adj) / (voc * isc), 3)
    else:
        ff = 0.0

    # ── ΔP = (1 - P_defect / P_healthy) × 100 ───────────────────────────────
    p_defect = vmp_adj * imp_adj
    delta_p  = round((1 - p_defect / HEALTHY_POWER) * 100, 2) if HEALTHY_POWER > 0 else 100.0

    # curve_type already in result from env_validation
    curve_type = clf_result.get("curve_type", "Unknown")

    logger.info(
        "Electrical estimate | class=%s | FF=%.3f | dP=%.1f%% | curve=%s",
        clf_result.get("subcategory", "?"), ff, delta_p, curve_type
    )

    result.update({
        "ff":              ff,
        "delta_p":         delta_p,
        "vmp_adjusted":    vmp_adj,
        "imp_adjusted":    imp_adj,
        "electrical_note": "ESTIMATED — replace with real IV data when available",
    })

    return result


# ──────────────────────────────────────────────────────────────────────────────
# SMOKE TEST  (python electrical_estimator.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys, cv2
    sys.path.insert(0, ".")

    from modules.classifier               import classify_panel
    from modules.damage_area_IR           import analyze_ir_damage
    from modules.environmental_validation import validate_environment

    IR_PATH = r"C:\Users\riyasharma\Documents\Solar Project\final_IR\BackSheetDamage\BacksheetDamage(2).png"

    ir_array = cv2.imread(IR_PATH)
    if ir_array is None:
        print(f"[ERROR] Image not found: {IR_PATH}")
        sys.exit(1)

    # Step 1 — classifier
    clf = classify_panel(
        panel_id = 1,
        mode     = "ir",
        ir_array = ir_array,
        tmin     = 20.0,
        tmax     = 100.0,
    )

    # Step 2 — IR damage area
    clf = analyze_ir_damage(clf, ir_array=ir_array)

    # Step 3 — env validation
    clf = validate_environment(clf, irradiance=900, wind_speed=4.5)

    # Step 4 — electrical estimate
    result = estimate_electrical(clf, pre_severity="Moderate")

    # print (skip mask)
    printable = {k: v for k, v in result.items() if k != "ir_hotspot_mask"}
    print(json.dumps(printable, indent=2, default=str))

    print("\n========== ELECTRICAL ESTIMATE ==========")
    print(f"Class        : {result['main_category']} -> {result['subcategory']}")
    print(f"Curve Type   : {result['curve_type']}")
    print(f"Voc / Isc    : {result['voc']}V / {result['isc']}A")
    print(f"Vmp / Imp    : {result['vmp_adjusted']}V / {result['imp_adjusted']}A")
    print(f"Fill Factor  : {result['ff']}")
    print(f"Power Loss   : {result['delta_p']}%")
    print(f"Note         : {result['electrical_note']}")
    print("=========================================")