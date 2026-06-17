"""
environmental_validation.py
============================
Pipeline Position : Step 3 — after damage_area_IR.py

Input  : clf_result  — combined output of classifier.py + damage_area_IR.py
         irradiance  — W/m² (from drone/operator, default 1000)
         wind_speed  — m/s  (from drone/operator, default 2.0)

Internally calls:
    hko_correction.py      → get_hko_weather(), process_measurement()
    iv_curve_simulator.py  → simulate_iv_curve()

Output : clf_result extended with env + IV fields

Does NOT do:
    - Classification        → classifier.py
    - IR damage area        → damage_area_IR.py
    - Electrical estimation → electrical_estimator.py
    - Final severity        → severity.py

Usage
-----
>>> import cv2
>>> from classifier               import classify_panel
>>> from damage_area_IR           import analyze_ir_damage
>>> from environmental_validation import validate_environment

>>> ir_array = cv2.imread("panel_ir.png")
>>> clf      = classify_panel(1, "ir", ir_array=ir_array, tmin=20, tmax=100)
>>> clf      = analyze_ir_damage(clf, ir_array=ir_array)
>>> result   = validate_environment(clf, irradiance=900, wind_speed=4.5)

New keys added to output dict:
{
    "env_status"        : "ACCEPTED",   # ACCEPTED / REJECTED / SKIPPED
    "delta_t_corrected" : 82.3,         # None if rejected / skipped
    "humidity"          : 72,
    "ambient_temp"      : 31,
    "irradiance"        : 900,
    "wind_speed"        : 4.5,
    "curve_type"        : "Flattened",
    "voc"               : 38.5,
    "isc"               : 6.5,
    "vmp"               : 30.5,
    "imp"               : 6.75,
    "fill_factor"       : 0.612,
    "power"             : 205.88,
    "power_loss"        : 19.58,
}
"""

from __future__ import annotations

import logging

from hko_correction     import get_hko_weather, process_measurement
from iv_curve_simulator import simulate_iv_curve

logger = logging.getLogger(__name__)

DEFAULT_IRRADIANCE = 1000
DEFAULT_WIND_SPEED = 2.0

# ──────────────────────────────────────────────────────────────────────────────
# SUBCATEGORY → iv_curve_simulator DEFECT_DATA key
# ──────────────────────────────────────────────────────────────────────────────

SUBCAT_TO_IV: dict = {
    "Snail Trail / Microcrack":                       "SnailTrails_Microcracks",
    "Hot Spots":                                      "Hotspot",
    "Hot Spots – Soiling/Shading (Hard, Bird Poop)":  "Hotspot_HardShading",
    "Soft Shading / Soiling":                         "Hotspot_SoftShading",
    "Junction Box Fault":                             "BypassDiode_Open",
    "Junction Box (Open)":                            "BypassDiode_Open",
    "Junction Box (Short)":                           "BypassDiode_Short",
    "Backsheet Image":                                "BacksheetDamage",
    "EVA Delamination / Discoloration / Corrosion":   "EVA_Delamination",
    "PID":                                            "PID",
    "Inverter / Battery Damage":                      "InverterBatteryDamage",
    "Glass Break":                                    "GlassBreak",
    "Full Panel Unresponsive":                        "FullPanelUnresponsive",
    "Unusual Cooling / Full Panel Down":              "UnusualCooling",
    "No Defect":                                      "Healthy",
}


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def validate_environment(
    clf_result: dict,
    irradiance: float = DEFAULT_IRRADIANCE,
    wind_speed: float = DEFAULT_WIND_SPEED,
) -> dict:
    """
    Run HKO environmental correction + IV curve simulation.

    Parameters
    ----------
    clf_result : dict  — output of analyze_ir_damage()
                         must have: subcategory, delta_t_proxy
    irradiance : float — site irradiance W/m²
    wind_speed : float — site wind speed m/s

    Returns
    -------
    dict — all clf_result keys + env/IV keys
    """
    result        = dict(clf_result)
    delta_t_proxy = clf_result.get("delta_t_proxy")

    if delta_t_proxy is None:
        logger.warning("delta_t_proxy missing — env correction skipped")
        result.update(_skipped(irradiance, wind_speed))
        return result

    # ── HKO weather ──────────────────────────────────────────────────────────
    try:
        humidity, ambient_temp = get_hko_weather()
    except Exception as exc:
        logger.warning("HKO API failed: %s — using defaults (70%%, 30C)", exc)
        humidity, ambient_temp = 70, 30

    # ── IV curve simulation ───────────────────────────────────────────────────
    subcategory = clf_result.get("subcategory", "")
    iv_key      = SUBCAT_TO_IV.get(subcategory, "Healthy")

    try:
        iv = simulate_iv_curve(iv_key)
    except Exception as exc:
        logger.warning("IV simulation failed: %s — using Healthy", exc)
        iv = simulate_iv_curve("Healthy")

    # ── HKO correction ────────────────────────────────────────────────────────
    try:
        correction = process_measurement(
            delta_t    = delta_t_proxy,
            irradiance = irradiance,
            wind_speed = wind_speed,
            humidity   = humidity,
            delta_p    = iv.power_loss,
        )
    except Exception as exc:
        logger.error("process_measurement failed: %s", exc)
        correction = {"status": "SKIPPED", "delta_t_corrected": None}

    delta_t_corrected = correction.get("delta_t_corrected")
    env_status        = correction.get("status", "SKIPPED")

    result.update({
        # env correction
        "env_status":         env_status,
        "delta_t_corrected":  round(float(delta_t_corrected), 2)
                              if delta_t_corrected is not None else None,
        # weather
        "humidity":           humidity,
        "ambient_temp":       ambient_temp,
        "irradiance":         irradiance,
        "wind_speed":         wind_speed,
        # IV curve
        "curve_type":         iv.curve_type,
        "voc":                iv.voc,
        "isc":                iv.isc,
        "vmp":                iv.vmp,
        "imp":                iv.imp,
        "fill_factor":        iv.fill_factor,
        "power":              iv.power,
        "power_loss":         iv.power_loss,
    })

    return result


# ──────────────────────────────────────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────────────────────────────────────

def _skipped(irradiance: float, wind_speed: float) -> dict:
    return {
        "env_status":        "SKIPPED",
        "delta_t_corrected": None,
        "humidity":          None,
        "ambient_temp":      None,
        "irradiance":        irradiance,
        "wind_speed":        wind_speed,
        "curve_type":        None,
        "voc":               None,
        "isc":               None,
        "vmp":               None,
        "imp":               None,
        "fill_factor":       None,
        "power":             None,
        "power_loss":        None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SMOKE TEST  (python environmental_validation.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys, cv2
    sys.path.insert(0, ".")

    from classifier     import classify_panel
    from damage_area_IR import analyze_ir_damage

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

    # Step 3 — environmental validation
    result = validate_environment(clf, irradiance=900, wind_speed=4.5)

    # print (skip mask)
    printable = {k: v for k, v in result.items() if k != "ir_hotspot_mask"}
    print(json.dumps(printable, indent=2, default=str))

    print("\n========== ENV VALIDATION ==========")
    print(f"Class            : {result['main_category']} -> {result['subcategory']}")
    print(f"Delta T Proxy    : {result['delta_t_proxy']}")
    print(f"Delta T Corrected: {result['delta_t_corrected']}")
    print(f"Env Status       : {result['env_status']}")
    print(f"Humidity         : {result['humidity']}%")
    print(f"Ambient Temp     : {result['ambient_temp']} C")
    print(f"Irradiance       : {result['irradiance']} W/m2")
    print(f"Wind Speed       : {result['wind_speed']} m/s")
    print(f"Curve Type       : {result['curve_type']}")
    print(f"Fill Factor      : {result['fill_factor']}")
    print(f"Power Loss       : {result['power_loss']}%")
    print("=====================================")