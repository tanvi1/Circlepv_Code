import numpy as np
import requests

# =====================================================
# HKO WEATHER
# =====================================================

def get_hko_weather():

    try:

        url = (
            "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
            "?dataType=rhrread&lang=en"
        )

        data = requests.get(
            url,
            timeout=10
        ).json()

        humidity = data["humidity"]["data"][0]["value"]

        temperature = data["temperature"]["data"][0]["value"]

        return humidity, temperature

    except Exception as e:

        print(
            f"[WARNING] Weather API failed: {e}"
        )

        return 70, 30


# =====================================================
# FACTORS
# =====================================================

def MG(irradiance):
    return irradiance / 1000.0


def Mv(wind_speed, k):

    if wind_speed <= 3:
        return 1.0

    return 1 + k * (wind_speed - 3)


def MRH(humidity, alpha):

    if humidity <= 60:
        return 1.0

    return max(
        1 - alpha * ((humidity - 60) / 40),
        0.1
    )


# =====================================================
# DELTA T CORRECTION
# =====================================================

def corrected_delta_t(
        delta_t,
        irradiance,
        wind_speed,
        humidity,
        k=0.10,
        alpha=0.93):

    return (
        delta_t
        * MG(irradiance)
        * Mv(wind_speed, k)
        * MRH(humidity, alpha)
    )


# =====================================================
# OPTIMIZATION
# =====================================================

def optimize_delta_t(
        delta_t,
        delta_p,
        irradiance,
        wind_speed,
        humidity):

    best_J = float("inf")

    best_dt = None
    best_k = None
    best_alpha = None

    for k in np.arange(0.05, 0.151, 0.01):

        k = float(k)

        for alpha in np.arange(0.91, 0.951, 0.01):

            alpha = float(alpha)

            dt_corr = corrected_delta_t(
                delta_t,
                irradiance,
                wind_speed,
                humidity,
                k,
                alpha
            )

            J = abs(
                float(dt_corr)
                - float(delta_p)
            )

            if J < best_J:

                best_J = J
                best_dt = dt_corr
                best_k = k
                best_alpha = alpha

    return {

        "delta_t_corrected":
            round(float(best_dt), 2),

        "delta_p":
            round(float(delta_p), 2),

        "J":
            round(float(best_J), 2),

        "k":
            round(float(best_k), 3),

        "alpha":
            round(float(best_alpha), 3)
    }


# =====================================================
# MAIN
# =====================================================

def process_measurement(
        delta_t,
        irradiance,
        wind_speed,
        humidity,
        delta_p=None):

    EPSILON = 5.0

    # ====================================
    # INVALID CONDITIONS
    # ====================================

    if (
        irradiance < 500
        or wind_speed > 8
        or humidity > 90
    ):

        return {

            "status":
                "REJECTED",

            "reason":
                "Invalid conditions",

            "irradiance":
                irradiance,

            "wind_speed":
                wind_speed,

            "humidity":
                humidity
        }

    # ====================================
    # NO DELTA P AVAILABLE
    # ====================================

    if delta_p is None:

        dt_corr = corrected_delta_t(
            delta_t,
            irradiance,
            wind_speed,
            humidity
        )

        return {

            "status":
                "ACCEPTED",

            "mode":
                "NO_DELTA_P",

            "delta_t_corrected":
                round(float(dt_corr), 2)
        }

    # ====================================
    # DIRECT MODE
    # ====================================

    if (
        irradiance >= 600
        and wind_speed <= 7
        and humidity <= 85
    ):

        dt_corr = corrected_delta_t(
            delta_t,
            irradiance,
            wind_speed,
            humidity
        )

        J = abs(
            float(dt_corr)
            - float(delta_p)
        )

        return {

            "status":
                "ACCEPTED",

            "mode":
                "DIRECT",

            "delta_t_corrected":
                round(float(dt_corr), 2),

            "delta_p":
                round(float(delta_p), 2),

            "J":
                round(float(J), 2)
        }

    # ====================================
    # OPTIMIZATION MODE
    # ====================================

    result = optimize_delta_t(
        delta_t,
        delta_p,
        irradiance,
        wind_speed,
        humidity
    )

    if result["J"] <= EPSILON:

        result["status"] = "ACCEPTED"
        result["mode"] = "OPTIMIZED"
        result["threshold"] = EPSILON

    else:

        result["status"] = "REJECTED"
        result["mode"] = "OPTIMIZED_FAILED"
        result["threshold"] = EPSILON

        result["reason"] = (
            f"J={result['J']} > "
            f"epsilon={EPSILON}"
        )

    return result