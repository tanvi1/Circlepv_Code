# iv_curve_simulator.py

from dataclasses import dataclass


# =====================================================
# HEALTHY REFERENCE PANEL
# =====================================================

HEALTHY_VOC = 40.0
HEALTHY_ISC = 8.0

HEALTHY_VMP = 32.0
HEALTHY_IMP = 8.0

HEALTHY_POWER = HEALTHY_VMP * HEALTHY_IMP


# =====================================================
# RESULT CLASS
# =====================================================

@dataclass
class IVCurveResult:

    curve_type: str

    voc: float
    isc: float

    vmp: float
    imp: float

    fill_factor: float

    power: float

    power_loss: float


# =====================================================
# CLIENT TABLE
# =====================================================

DEFECT_DATA = {

    "SnailTrails_Microcracks": {
        "curve_type": "Flattened",
        "voc": 38.5,
        "isc": 6.5,
        "vmp": 30.5,
        "imp": 6.75
    },

    "Hotspot": {
        "curve_type": "Truncated",
        "voc": 29.0,
        "isc": 5.0,
        "vmp": 27.0,
        "imp": 5.5
    },

    "Hotspot_HardShading": {
        "curve_type": "Distorted",
        "voc": 31.0,
        "isc": 5.5,
        "vmp": 29.0,
        "imp": 5.75
    },

    "Hotspot_SoftShading": {
        "curve_type": "Flattened",
        "voc": 38.5,
        "isc": 6.5,
        "vmp": 30.5,
        "imp": 6.75
    },

    "BypassDiode_Open": {
        "curve_type": "Distorted",
        "voc": 36.0,
        "isc": 6.5,
        "vmp": 30.5,
        "imp": 6.75
    },

    "BypassDiode_Short": {
        "curve_type": "Distorted",
        "voc": 32.5,
        "isc": 6.5,
        "vmp": 28.0,
        "imp": 6.25
    },

    "BacksheetDamage": {
        "curve_type": "Flattened",
        "voc": 38.5,
        "isc": 6.5,
        "vmp": 30.5,
        "imp": 6.75
    },

    "EVA_Delamination": {
        "curve_type": "Flattened",
        "voc": 38.5,
        "isc": 6.5,
        "vmp": 30.5,
        "imp": 6.75
    },

    "PID": {
        "curve_type": "Flattened",
        "voc": 38.5,
        "isc": 6.5,
        "vmp": 29.0,
        "imp": 6.25
    },

    "InverterBatteryDamage": {
        "curve_type": "Truncated",
        "voc": 32.5,
        "isc": 6.5,
        "vmp": 28.0,
        "imp": 6.25
    },

    "GlassBreak": {
        "curve_type": "Flattened",
        "voc": 38.5,
        "isc": 6.5,
        "vmp": 30.5,
        "imp": 6.75
    },

    "FullPanelUnresponsive": {
        "curve_type": "FlatLine",
        "voc": 40.0,
        "isc": 0.0,
        "vmp": 0.0,
        "imp": 0.0
    },

    "UnusualCooling": {
        "curve_type": "FlatLine",
        "voc": 40.0,
        "isc": 0.0,
        "vmp": 0.0,
        "imp": 0.0
    },

    "Healthy": {
        "curve_type": "Normal",
        "voc": 40.0,
        "isc": 8.0,
        "vmp": 32.0,
        "imp": 8.0
    }
}


# =====================================================
# CALCULATE FF
# =====================================================

def calculate_ff(voc, isc, vmp, imp):

    denominator = voc * isc

    if denominator <= 0:
        return 0.0

    return (vmp * imp) / denominator


# =====================================================
# CALCULATE POWER LOSS
# =====================================================

def calculate_power_loss(vmp, imp):

    power = vmp * imp

    power_loss = (
        (HEALTHY_POWER - power)
        / HEALTHY_POWER
    ) * 100

    return power, max(power_loss, 0)


# =====================================================
# MAIN FUNCTION
# =====================================================

def simulate_iv_curve(defect_name):

    if defect_name not in DEFECT_DATA:
        defect_name = "Healthy"

    d = DEFECT_DATA[defect_name]

    ff = calculate_ff(
        d["voc"],
        d["isc"],
        d["vmp"],
        d["imp"]
    )

    power, power_loss = calculate_power_loss(
        d["vmp"],
        d["imp"]
    )

    return IVCurveResult(

        curve_type=d["curve_type"],

        voc=d["voc"],
        isc=d["isc"],

        vmp=d["vmp"],
        imp=d["imp"],

        fill_factor=round(ff, 3),

        power=round(power, 2),

        power_loss=round(power_loss, 2)
    )


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    result = simulate_iv_curve(
        "Hotspot"
    )

    print()

    print("Curve Type :", result.curve_type)

    print("Voc        :", result.voc)

    print("Isc        :", result.isc)

    print("Vmp        :", result.vmp)

    print("Imp        :", result.imp)

    print("FF         :", result.fill_factor)

    print("Power      :", result.power)

    print("Power Loss :", result.power_loss)