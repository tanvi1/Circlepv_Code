"""
Solar Panel Damage Detection — Classical CV
Supports: GlassBreak | HardShading | SnailTrails_Microcracks

Usage:
    result = analyze_rgb_damage("image.jpg", "GlassBreak")
    result = analyze_rgb_damage(img_array,  "HardShading")
    result = analyze_rgb_damage("image.jpg", "SnailTrails_Microcracks")
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import os


# ══════════════════════════════════════════════
# Standard result format — same for ALL classes
# ══════════════════════════════════════════════
@dataclass
class DamageResult:
    class_name: str
    detected: bool
    confidence: float             # 0.0 – 1.0
    damage_percentage: float      # % of panel area affected
    damage_mask: Optional[np.ndarray] = None   # binary mask (H×W uint8)
    contours: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    annotated_image: Optional[np.ndarray] = None


# ══════════════════════════════════════════════════════════════════
#  CLASS 1: GLASS BREAK
#  Visual pattern: spider-web / radial cracks from impact point
#  Detection:  bright pixel mask + local variance → impact point
# ══════════════════════════════════════════════════════════════════

def _gb_preprocess(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = lab[:,:,0]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return gray, clahe.apply(gray)


def _gb_crack_masks(enhanced):
    thresh = max(int(np.percentile(enhanced, 75)), 140)
    _, bright = cv2.threshold(enhanced, thresh, 255, cv2.THRESH_BINARY)

    bg = cv2.GaussianBlur(enhanced.astype(np.float32), (15, 15), 0)
    var = (enhanced.astype(np.float32) - bg) ** 2
    var_norm = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, var_mask = cv2.threshold(var_norm, 25, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_or(bright, var_mask)
    return combined, float(np.sum(bright > 0)) / bright.size


def _gb_refine(combined):
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    closed  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close, iterations=2)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open, iterations=1)


def _gb_impact_point(enhanced, mask):
    region = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    blob   = cv2.GaussianBlur(region.astype(np.float32), (31, 31), 0)
    _, max_val, _, max_loc = cv2.minMaxLoc(blob)
    return max_loc if max_val >= 50 else None


def _gb_contours(mask, total_area):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid, boxes, area = [], [], 0
    min_t = total_area * 0.0005
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_t:
            continue
        valid.append(c)
        boxes.append(cv2.boundingRect(c))
        area += a
    return valid, boxes, round(100.0 * area / total_area, 3)


def _gb_confidence(bright_ratio, dmg_pct, n_cnts, has_impact):
    s = 0.0
    s += 0.25 if bright_ratio >= 0.15 else (0.12 if bright_ratio >= 0.05 else 0.0)
    s += 0.35 if dmg_pct >= 30 else (0.20 if dmg_pct >= 5 else (0.08 if dmg_pct > 0 else 0.0))
    s += 0.20 if n_cnts >= 1 else 0.0
    s += 0.20 if has_impact else 0.0
    return min(round(s, 2), 1.0)


def _gb_annotate(image, contours, boxes, impact, dmg_pct, conf):
    out = image.copy()
    overlay = out.copy()
    cv2.drawContours(overlay, contours, -1, (0, 0, 200), thickness=cv2.FILLED)
    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(out, (x, y), (x+bw, y+bh), (0, 220, 220), 1)
    if impact:
        cx, cy = impact
        cv2.circle(out, (cx, cy), 18, (255, 200, 0), 2)
        cv2.circle(out, (cx, cy), 3,  (255, 200, 0), -1)
    #     cv2.putText(out, "impact", (cx+22, cy+5),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
    # cv2.rectangle(out, (0,0), (out.shape[1], 34), (20,20,20), -1)
    # cv2.putText(out, f"GlassBreak | dmg:{dmg_pct:.1f}% | conf:{conf:.2f}",
    #             (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,80,255), 2)
    return out


def detect_glass_break(image: np.ndarray) -> DamageResult:
    h, w   = image.shape[:2]
    total  = h * w
    _, enh = _gb_preprocess(image)
    comb, bright_ratio = _gb_crack_masks(enh)
    refined = _gb_refine(comb)
    impact  = _gb_impact_point(enh, refined)
    cnts, boxes, dmg_pct = _gb_contours(refined, total)
    conf = _gb_confidence(bright_ratio, dmg_pct, len(cnts), impact is not None)
    detected = conf >= 0.35

    return DamageResult(
        class_name="GlassBreak",
        detected=detected,
        confidence=conf,
        damage_percentage=dmg_pct,
        damage_mask=refined,
        contours=cnts,
        metadata={
            "impact_point": impact,
            "bright_pixel_ratio": round(bright_ratio, 4),
            "contour_count": len(cnts),
            "bounding_boxes": boxes,
        },
        annotated_image=_gb_annotate(image, cnts, boxes, impact, dmg_pct, conf),
    )


# ══════════════════════════════════════════════════════════════════
#  CLASS 2: HARD SHADING
#  Visual pattern: vertical dark band (bypass diode shading) +
#                  bright blob (bypass diode damage point)
#  Detection:  column-wise intensity analysis + bright blob detection
# ══════════════════════════════════════════════════════════════════

def _hs_dark_bands(gray, threshold_ratio=0.82):
    """Find contiguous vertical columns that are significantly darker."""
    col_means = gray.mean(axis=0)
    overall   = col_means.mean()
    dark_flag = col_means < (overall * threshold_ratio)

    bands = []
    in_band, start = False, 0
    for i, flag in enumerate(dark_flag):
        if flag and not in_band:
            start, in_band = i, True
        elif not flag and in_band:
            bands.append((start, i))
            in_band = False
    if in_band:
        bands.append((start, len(dark_flag)))
    return bands, col_means, overall


def _hs_bright_blobs(gray):
    """Detect bright white blobs (bypass diode burn/bubble)."""
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 50:
            continue
        blobs.append((int(a), cv2.boundingRect(c), c))
    return blobs


def _hs_confidence(shade_pct, n_bands, n_blobs):
    s = 0.0
    s += 0.50 if shade_pct >= 20 else (0.30 if shade_pct >= 8 else (0.10 if shade_pct > 0 else 0.0))
    s += 0.20 if n_bands >= 1 else 0.0
    s += 0.30 if n_blobs >= 1 else 0.0
    return min(round(s, 2), 1.0)


def _hs_annotate(image, bands, blobs, shade_pct, conf):
    h, w = image.shape[:2]
    out = image.copy()
    # Dark band overlay — blue tint
    for (s, e) in bands:
        overlay = out.copy()
        overlay[:, s:e] = (180, 60, 0)
        # cv2.addWeighted(overlay, 0.30, out, 0.70, 0, out)
        # cv2.rectangle(out, (s, 0), (e, h), (255, 100, 0), 2)
    # Bright blobs — yellow
    for (_, bbox, cnt) in blobs:
        x, y, bw, bh = bbox
        # cv2.drawContours(out, [cnt], -1, (255, 0, 0), 2)
        # cv2.putText(out, "diode", (x, y-5),
        #
        # cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    # cv2.rectangle(out, (0,0), (out.shape[1], 34), (20,20,20), -1)
    # cv2.putText(out, f"HardShading | dmg:{shade_pct:.1f}% | conf:{conf:.2f}",
    #             (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 0), 2)
    return out


def detect_hard_shading(image: np.ndarray) -> DamageResult:
    h, w  = image.shape[:2]
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bands, col_means, overall = _hs_dark_bands(gray)
    blobs = _hs_bright_blobs(gray)

    # Shade mask — all dark band columns
    shade_mask = np.zeros((h, w), dtype=np.uint8)
    shade_area = 0
    for (s, e) in bands:
        if (e - s) > w * 0.02:   # band must be >2% of width
            shade_mask[:, s:e] = 255
            shade_area += (e - s) * h

    shade_pct = round(100.0 * shade_area / (h * w), 3)
    valid_bands = [(s,e) for s,e in bands if (e-s) > w*0.02]

    conf = _hs_confidence(shade_pct, len(valid_bands), len(blobs))
    detected = conf >= 0.30

    return DamageResult(
        class_name="HardShading",
        detected=detected,
        confidence=conf,
        damage_percentage=shade_pct,
        damage_mask=shade_mask,
        metadata={
            "dark_bands": valid_bands,
            "n_bands": len(valid_bands),
            "bright_blobs": len(blobs),
            "col_mean_overall": round(float(overall), 1),
            "col_mean_min": round(float(col_means.min()), 1),
        },
        annotated_image=_hs_annotate(image, valid_bands, blobs, shade_pct, conf),
    )


# ══════════════════════════════════════════════════════════════════
#  CLASS 3: SNAIL TRAILS / MICROCRACKS  (v2 — tight boundary edition)
#
#  Visual signatures detected:
#    • Microcrack    → thin bright/white diagonal crack line on dark panel
#    • Snail trail   → brownish discoloration tracing cell boundaries
#
#  Pipeline (per function):
#    _mc_enhance()       — CLAHE + background subtraction → diff image
#    _mc_diagonal_mask() — Hough lines, keep ONLY diagonal angles,
#                          intersect with bright-diff pixels
#    _mc_crack_contours()— grow mask → find contours → filter by
#                          min-area + elongation ratio
#    _mc_confidence()    — score based on crack count + damage %
#    _mc_annotate()      — tight red contour + cyan rotated bounding rect
#                          drawn per crack region (no full-image lines)
#
#  Pipeline position:  RGB_Damage_area.py → CLASS_REGISTRY
#  Inputs:  BGR np.ndarray (single cropped panel image)
#  Outputs: DamageResult
#    .detected            bool
#    .confidence          float 0–1
#    .damage_percentage   float %  (crack pixels / total panel pixels)
#    .damage_mask         uint8 binary mask  (255 = crack pixel)
#    .contours            list of crack contours (tight boundary per crack)
#    .annotated_image     BGR with contours + rotated bounding rects drawn
#    .metadata:
#        crack_count      int   number of valid crack regions
#        crack_segments   list[(x1,y1,x2,y2)]  Hough line segments kept
#        elongation_list  list[float]  per-crack elongation ratio
#        bounding_rects   list[(cx,cy,w,h,angle)]  rotated bounding rects
#
#  Future integration:
#    • Severity module reads .damage_percentage + .confidence
#    • YOLO fusion: replace _mc_crack_contours() with YOLO masks when available
#    • Cell-level analysis: run per panel-crop from panel_cell_detection.py
# ══════════════════════════════════════════════════════════════════

# ── Tuneable constants (adjust per deployment / panel type) ───────
_MC_CLAHE_CLIP      = 4.0    # CLAHE clip limit — higher = more local contrast
_MC_CLAHE_TILE      = (8, 8) # CLAHE tile grid
_MC_BG_BLUR_K       = 21     # Gaussian kernel for background estimation
_MC_DIFF_PERCENTILE = 85     # Threshold: keep brightest N% of diff pixels
_MC_HOUGH_THRESH    = 8      # Hough accumulator threshold (lower = more lines)
_MC_HOUGH_GAP       = 8      # Max gap (px) to join Hough line segments
_MC_H_ANGLE         = 15     # Degrees — angles < this are "horizontal" (grid)
_MC_V_ANGLE_LO      = 75     # Degrees — angles between LO–HI are "vertical"
_MC_V_ANGLE_HI      = 105
_MC_LINE_DILATE_K   = 5      # Kernel to broaden Hough lines before AND
_MC_LINE_DILATE_I   = 2      # Dilation iterations for Hough line broadening
_MC_GROW_K          = 3      # Kernel to grow crack pixels into tight regions
_MC_GROW_I          = 3      # Growth iterations
_MC_MIN_AREA_FRAC   = 0.0008 # Min crack contour area as fraction of panel
_MC_MIN_ELONGATION  = 2    # Min elongation ratio (max/min side of rotated rect)


def _mc_enhance(image: np.ndarray) -> tuple:
    """
    CLAHE enhancement + large-scale background subtraction.

    Returns
    -------
    enhanced : uint8 gray  — CLAHE-enhanced grayscale
    diff     : uint8 gray  — bright residual after background removal
                             high values = locally brighter than surroundings
    """
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=_MC_CLAHE_CLIP,
                               tileGridSize=_MC_CLAHE_TILE)
    enhanced = clahe.apply(gray)
    bg       = cv2.GaussianBlur(enhanced.astype(np.float32),
                                (_MC_BG_BLUR_K, _MC_BG_BLUR_K), 0)
    diff     = np.clip(enhanced.astype(np.float32) - bg, 0, 255).astype(np.uint8)
    return enhanced, diff


def _mc_diagonal_mask(diff: np.ndarray, image_shape: tuple) -> tuple:
    """
    Build a binary mask containing ONLY diagonal crack pixels.

    Strategy
    --------
    1. Threshold diff at adaptive percentile → bright candidate pixels.
    2. Hough probabilistic line detection on that mask.
    3. Keep only lines whose angle is NOT horizontal (grid) or vertical (grid).
    4. Dilate kept lines and AND with bright-pixel mask.
       → Result: pixels that are BOTH bright AND lie on a diagonal line.

    Returns
    -------
    crack_pixels : uint8 binary mask
    crack_segs   : list[(x1,y1,x2,y2)] — raw diagonal Hough segments
    """
    h, w      = image_shape[:2]
    th_val    = int(np.percentile(diff, _MC_DIFF_PERCENTILE))
    _, bright = cv2.threshold(diff, th_val, 255, cv2.THRESH_BINARY)

    min_line_len = int(min(h, w) * 0.05)
    lines = cv2.HoughLinesP(
        bright, 1, np.pi / 180,
        threshold=_MC_HOUGH_THRESH,
        minLineLength=min_line_len,
        maxLineGap=_MC_HOUGH_GAP,
    )

    line_mask  = np.zeros((h, w), dtype=np.uint8)
    crack_segs = []

    if lines is not None:
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            angle = abs(np.degrees(
                np.arctan2(float(y2 - y1), float(x2 - x1) + 1e-5)
            ))
            is_horizontal = angle < _MC_H_ANGLE or angle > (180 - _MC_H_ANGLE)
            is_vertical   = _MC_V_ANGLE_LO < angle < _MC_V_ANGLE_HI
            if not (is_horizontal or is_vertical):          # diagonal → crack
                crack_segs.append((x1, y1, x2, y2))
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

    # Broaden lines then restrict to bright pixels only
    dilate_k     = np.ones((_MC_LINE_DILATE_K, _MC_LINE_DILATE_K), np.uint8)
    broad_lines  = cv2.dilate(line_mask, dilate_k, iterations=_MC_LINE_DILATE_I)
    crack_pixels = cv2.bitwise_and(bright, broad_lines)
    return crack_pixels, crack_segs


def _mc_crack_contours(crack_pixels: np.ndarray, image_shape: tuple) -> tuple:
    """
    Grow crack pixels slightly, find contours, filter by area + elongation.

    Elongation filter:  crack regions are narrow and long.
    Blobs that are roughly square (elongation < _MC_MIN_ELONGATION) are
    rejected as grid-intersection artefacts or noise.

    Returns
    -------
    valid_contours  : list of np.ndarray contours  (tight crack boundaries)
    crack_mask      : uint8 binary mask of accepted crack regions
    bounding_rects  : list[(cx, cy, rw, rh, angle)]  rotated bounding rects
    elongation_list : list[float]  elongation per contour
    """
    h, w      = image_shape[:2]
    grow_k    = np.ones((_MC_GROW_K, _MC_GROW_K), np.uint8)
    grown     = cv2.dilate(crack_pixels, grow_k, iterations=_MC_GROW_I)

    cnts, _   = cv2.findContours(grown, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    min_area  = h * w * _MC_MIN_AREA_FRAC

    valid, crack_mask = [], np.zeros((h, w), dtype=np.uint8)
    bounding_rects, elongation_list = [], []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        rect       = cv2.minAreaRect(c)                # ((cx,cy),(rw,rh), angle)
        (cx, cy), (rw, rh), angle = rect
        if rw < 1 or rh < 1:
            continue
        elongation = max(rw, rh) / min(rw, rh)

        if elongation < _MC_MIN_ELONGATION:             # too blobby → skip
            continue

        valid.append(c)
        cv2.drawContours(crack_mask, [c], -1, 255, cv2.FILLED)
        bounding_rects.append((round(cx, 1), round(cy, 1),
                               round(rw, 1), round(rh, 1), round(angle, 1)))
        elongation_list.append(round(elongation, 2))

    return valid, crack_mask, bounding_rects, elongation_list


def _mc_confidence(n_cracks: int, dmg_pct: float, n_segs: int) -> float:
    """
    Score 0–1 based on:
      • Number of confirmed crack contours  (weight 0.40)
      • Damage percentage                   (weight 0.40)
      • Supporting Hough segments           (weight 0.20)
    """
    s = 0.0
    # Crack contour count
    if   n_cracks >= 5:  s += 0.40
    elif n_cracks >= 2:  s += 0.30
    elif n_cracks >= 1:  s += 0.20
    # Damage area
    if   dmg_pct >= 10:  s += 0.40
    elif dmg_pct >= 3:   s += 0.25
    elif dmg_pct > 0:    s += 0.10
    # Supporting segments
    if   n_segs >= 15:   s += 0.20
    elif n_segs >= 5:    s += 0.12
    elif n_segs >= 1:    s += 0.06
    return min(round(s, 2), 1.0)


def _mc_annotate(
    image: np.ndarray,
    contours: list,
    bounding_rects: list,
    dmg_pct: float,
    conf: float,
) -> np.ndarray:
    """
    Draw tight crack boundaries on a copy of the original image.

    Layers (bottom → top):
      1. Semi-transparent red fill inside each crack contour
      2. Solid red outline  (2 px) — tight contour boundary
      3. Cyan rotated bounding rect (1 px) per crack
      4. Dark HUD bar at top with stats
    """
    out     = image.copy()
    overlay = out.copy()

    # Layer 1 — red fill
    cv2.drawContours(overlay, contours, -1, (0, 0, 220), cv2.FILLED)
    cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)

    # Layer 2 — tight red contour border
    # cv2.drawContours(out, contours, -1, (0, 0, 255), 2)

    # Layer 3 — cyan rotated bounding rect (shows elongation axis)
    for (cx, cy, rw, rh, angle) in bounding_rects:
        rect_pts = cv2.boxPoints(((cx, cy), (rw, rh), angle))
        rect_pts = rect_pts.astype(np.int32)
        # cv2.drawContours(out, [rect_pts], 0, (0, 220, 255), 1)

    # Layer 4 — HUD
    # cv2.rectangle(out, (0, 0), (out.shape[1], 42), (15, 15, 15), -1)
    # cv2.putText(
    #     out,
    #     f"Microcrack / Snail Trail  |  Damage: {dmg_pct:.2f}%  "
    #     f"|  Cracks: {len(contours)}  |  Conf: {conf:.2f}",
    #     (8, 27),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.55,
    #     (0, 200, 255),
    #     2,
    # )
    return out
from skimage.filters import frangi

def detect_snail_trails_microcracks(image: np.ndarray) -> DamageResult:

    h, w = image.shape[:2]
    total_pixels = h * w

    # ==========================================================
    # PHASE 1 : FFT GRID REMOVAL
    # ==========================================================
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols, 2), np.uint8)

    cross_width = 3

    mask[crow-cross_width:crow+cross_width, :] = 0
    mask[:, ccol-cross_width:ccol+cross_width] = 0

    mask[crow-15:crow+15, ccol-15:ccol+15] = 1

    fshift = dft_shift * mask

    f_ishift = np.fft.ifftshift(fshift)

    img_back = cv2.idft(f_ishift)

    img_back = cv2.magnitude(
        img_back[:, :, 0],
        img_back[:, :, 1]
    )

    fft_cleaned = cv2.normalize(
        img_back,
        None,
        0,
        1.0,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_32F
    )

    # ==========================================================
    # PHASE 2 : DUAL POLARITY FRANGI
    # ==========================================================
    frangi_dark = frangi(
        fft_cleaned,
        sigmas=range(1, 4),
        black_ridges=True
    )

    frangi_light = frangi(
        fft_cleaned,
        sigmas=range(1, 4),
        black_ridges=False
    )

    frangi_combined = np.maximum(
        frangi_dark,
        frangi_light
    )

    frangi_output = cv2.normalize(
        frangi_combined,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # ==========================================================
    # PHASE 3 : THRESHOLD + CONTOURS
    # ==========================================================

    _, binary = cv2.threshold(
        frangi_output,
        35,     # tune if needed
        255,
        cv2.THRESH_BINARY
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    crack_mask = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        crack_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = [
        c for c in contours
        if cv2.contourArea(c) > 10
    ]

    final_mask = np.zeros_like(crack_mask)

    cv2.drawContours(
        final_mask,
        contours,
        -1,
        255,
        thickness=cv2.FILLED
    )

    # ==========================================================
    # PHASE 4 : METRICS
    # ==========================================================

    defect_pixels = cv2.countNonZero(final_mask)

    dmg_pct = round(
        (defect_pixels / total_pixels) * 100,
        4
    )

    crack_count = len(contours)

    confidence = min(
        1.0,
        (crack_count * 0.08) + (dmg_pct * 0.03)
    )

    detected = confidence >= 0.30

    # ==========================================================
    # ANNOTATION
    # ==========================================================

    annotated = image.copy()

    cv2.drawContours(
        annotated,
        contours,
        -1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated,
        f"Damage:{dmg_pct:.4f}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    return DamageResult(
        class_name="SnailTrails_Microcracks",
        detected=detected,
        confidence=round(confidence, 3),
        damage_percentage=dmg_pct,
        damage_mask=final_mask,
        contours=contours,
        metadata={
            "crack_count": crack_count,
            "method": "FFT + Dual Frangi"
        },
        annotated_image=annotated
    )

# def detect_snail_trails_microcracks(image: np.ndarray) -> DamageResult:
#     """
#     Detect microcracks and snail trails in a solar panel RGB image.

#     Uses background subtraction + diagonal Hough filtering to isolate
#     crack pixels, then draws tight contour boundaries around each crack
#     region rather than drawing across the full image.

#     Pipeline position
#     -----------------
#     Called by analyze_rgb_damage() via CLASS_REGISTRY.
#     Input is a single cropped panel BGR image (from panel_cell_detection.py).

#     Parameters
#     ----------
#     image : np.ndarray
#         BGR image of a single solar panel (uint8, any resolution).

#     Returns
#     -------
#     DamageResult
#         .class_name         "SnailTrails_Microcracks"
#         .detected           True if confidence >= 0.30
#         .confidence         0.0 – 1.0
#         .damage_percentage  % of panel covered by accepted crack contours
#         .damage_mask        uint8 binary mask  (255 = crack region)
#         .contours           list of tight contours per crack
#         .metadata
#             crack_count     int
#             crack_segments  list[(x1,y1,x2,y2)]  diagonal Hough segs
#             elongation_list list[float]
#             bounding_rects  list[(cx,cy,rw,rh,angle)]
#         .annotated_image    BGR with tight contour overlays
#     """
#     h, w  = image.shape[:2]
#     total = h * w

#     # Stage 1 — enhance + background subtract
#     _, diff = _mc_enhance(image)

#     # Stage 2 — isolate diagonal (crack) pixels via Hough filter
#     crack_pixels, crack_segs = _mc_diagonal_mask(diff, image.shape)

#     # Stage 3 — grow → contours → elongation filter
#     contours, crack_mask, bounding_rects, elongation_list = \
#         _mc_crack_contours(crack_pixels, image.shape)

#     # Stage 4 — metrics
#     crack_area = int((crack_mask > 0).sum())
#     dmg_pct    = round(100.0 * crack_area / total, 3)
#     conf       = _mc_confidence(len(contours), dmg_pct, len(crack_segs))
#     detected   = conf >= 0.30

#     return DamageResult(
#         class_name="SnailTrails_Microcracks",
#         detected=detected,
#         confidence=conf,
#         damage_percentage=dmg_pct,
#         damage_mask=crack_mask,
#         contours=contours,
#         metadata={
#             "crack_count":     len(contours),
#             "crack_segments":  crack_segs,
#             "elongation_list": elongation_list,
#             "bounding_rects":  bounding_rects,
#         },
#         annotated_image=_mc_annotate(
#             image, contours, bounding_rects, dmg_pct, conf
#         ),
#     )

def _ss_soiling_mask(image):
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H = hsv[:,:,0].astype(float)
    S = hsv[:,:,1].astype(float)
    V = hsv[:,:,2].astype(float)
    b = image[:,:,0].astype(np.float32)
    r = image[:,:,2].astype(np.float32)

    # Exclude sky / metal frame
    exclude = ((S < 15) & (V > 195)) | ((S < 10) & (V > 170))

    # Signal 1: Gray soiling — dust kills panel saturation
    gray_soil = (S < 60) & (V > 40) & (V < 215)

    # Signal 2: Brown/rust/organic dust
    brown_soil = (H >= 8) & (H <= 38) & (S > 15) & (V > 40)

    # Signal 3: R channel not dominated by B (panel is normally blue)
    not_blue = (r - b) > -15

    soiled = (gray_soil | brown_soil) & not_blue & ~exclude
    raw = soiled.astype(np.uint8) * 255

    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    cleaned = cv2.morphologyEx(raw,     cv2.MORPH_OPEN,  k_open,  iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=2)
    return cleaned


def _ss_pattern(mask):
    h = mask.shape[0]
    row_s = (mask > 0).mean(axis=1) * 100
    t  = float(row_s[:h//3].mean())
    m  = float(row_s[h//3:2*h//3].mean())
    bo = float(row_s[2*h//3:].mean())
    total = (mask > 0).mean() * 100

    max_z = max(t, m, bo)
    if   bo >= t * 1.3 and bo == max_z:   pat = "bottom_heavy"
    elif t  >= bo * 1.3 and t == max_z:   pat = "top_heavy"
    elif total > 40:                       pat = "uniform"
    else:                                  pat = "patches"
    return pat, round(t, 1), round(m, 1), round(bo, 1)


def _ss_confidence(soil_pct, n_patches):
    s = 0.0
    s += 0.60 if soil_pct >= 30 else (0.40 if soil_pct >= 10 else (0.15 if soil_pct > 0 else 0.0))
    s += 0.40 if n_patches >= 1 else 0.0
    return min(round(s, 2), 1.0)


def _ss_annotate(image, mask, contours, soil_pct, pattern, conf):
    out = image.copy()
    overlay = out.copy()
    overlay[mask > 0] = (30, 80, 200)
    cv2.addWeighted(overlay, 0.30, out, 0.70, 0, out)
    cv2.drawContours(out, contours, -1, (0, 60, 255), 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), (20, 20, 20), -1)
    # cv2.putText(out,
    #             f"SoftShading/Soiling | {pattern} | dmg:{soil_pct:.1f}% | conf:{conf:.2f}",
    #             (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 180, 255), 2)
    return out


def detect_soft_shading(image: np.ndarray) -> DamageResult:
    h, w  = image.shape[:2]
    total = h * w

    mask    = _ss_soiling_mask(image)
    pattern, top_pct, mid_pct, bot_pct = _ss_pattern(mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid   = [c for c in cnts if cv2.contourArea(c) > total * 0.003]

    soil_area = sum(cv2.contourArea(c) for c in valid)
    soil_pct  = round(100.0 * soil_area / total, 3)

    conf     = _ss_confidence(soil_pct, len(valid))
    detected = conf >= 0.30

    return DamageResult(
        class_name="SoftShading_Soiling",
        detected=detected,
        confidence=conf,
        damage_percentage=soil_pct,
        damage_mask=mask,
        contours=valid,
        metadata={
            "pattern":      pattern,          # uniform | patches | bottom_heavy | top_heavy
            "top_pct":      top_pct,
            "mid_pct":      mid_pct,
            "bottom_pct":   bot_pct,
            "n_patches":    len(valid),
        },
        annotated_image=_ss_annotate(image, mask, valid, soil_pct, pattern, conf),
    )
# ─────────────────────────────────────────────────────────────────
#  Shared drawing helper
# ─────────────────────────────────────────────────────────────────
def _draw_polygons(image: np.ndarray,
                   contours: list,
                   fill_color:   Tuple[int,int,int],
                   border_color: Tuple[int,int,int],
                   alpha: float = 0.30) -> np.ndarray:
    """
    Draw tight polygon overlays on a copy of the image.
    Each contour gets:
      • semi-transparent filled region
      • solid border (2 px)
      • cyan approxPolyDP polygon (tight outline, 1 px)
    """
    out     = image.copy()
    overlay = out.copy()
    cv2.drawContours(overlay, contours, -1, fill_color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
    # cv2.drawContours(out, contours, -1, border_color, 2)
    for c in contours:
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        cv2.polylines(out, [approx], True, (0, 220, 255), 1)
    return out


def _hud(image: np.ndarray, text: str) -> np.ndarray:
    # cv2.rectangle(image, (0, 0), (image.shape[1], 40), (15, 15, 15), -1)
    # cv2.putText(image, text, (8, 26),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 215, 255), 2)
    return image
def detect_backsheet_damage(image: np.ndarray) -> DamageResult:
    """
    Detect backsheet peeling / bubbling.
    Uses Canny edge detection → morphological close → large irregular contour.
    The peeled area appears as a bright irregular blob with strong edges.
    """
    h, w  = image.shape[:2]
    total = h * w
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges  = cv2.Canny(gray, 15, 45)
    k      = np.ones((11, 11), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=3)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid   = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < total * 0.02: continue               # too small
        p = cv2.arcLength(c, True)
        if p / max(a ** 0.5, 1) > 3.0:             # irregular (not simple rect)
            valid.append(c)

    dmg_pct = round(100.0 * sum(cv2.contourArea(c) for c in valid) / total, 3)
    s       = (0.60 if dmg_pct >= 20 else (0.40 if dmg_pct >= 5 else (0.20 if dmg_pct > 0 else 0.0)))
    s      += 0.40 if valid else 0.0
    conf    = min(round(s, 2), 1.0)
    detected= conf >= 0.35

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, valid, -1, 255, cv2.FILLED)

    out = _draw_polygons(image, valid, (100, 200, 255), (180, 220, 255))
    _hud(out, f"BacksheetDamage | dmg:{dmg_pct:.1f}% | conf:{conf:.2f} | regions:{len(valid)}")

    return DamageResult(
        class_name="BacksheetDamage", detected=detected,
        confidence=conf, damage_percentage=dmg_pct,
        damage_mask=mask, contours=valid,
        metadata={"region_count": len(valid)},
        annotated_image=out,
    )

# ══════════════════════════════════════════════════════════════════
#  UNIVERSAL ENTRY POINT
#  analyze_rgb_damage(image_input, class_name) → DamageResult
# ══════════════════════════════════════════════════════════════════

CLASS_REGISTRY = {
    "GlassBreak":                                       detect_glass_break,
    "HardShading":                                      detect_hard_shading,
    "SnailTrails_Microcracks":                          detect_snail_trails_microcracks,
    "SoftShading_Soiling":                              detect_soft_shading,
    "BacksheetDamage":                                  detect_backsheet_damage,
    
    # "Corrosion_Discoloration_Delamination":           detect_corrosion,
    # "HotSpots":           detect_hotspots,
    # "BypassDiode":        detect_bypass_diode,
    # "InverterBattery":    detect_inverter_battery,
}


def analyze_rgb_damage(image_input, class_name: str) -> DamageResult:
    """
    Universal entry point.

    Args:
        image_input : str (file path)  OR  np.ndarray (BGR)
        class_name  : "GlassBreak" | "HardShading" | "SnailTrails_Microcracks"
                      | "SoftShading_Soiling"

    Returns:
        DamageResult with:
            .detected           bool
            .confidence         float 0-1
            .damage_percentage  float %
            .damage_mask        np.ndarray binary
            .annotated_image    np.ndarray BGR
            .metadata           dict  (class-specific extras)

    Example:
        r = analyze_rgb_damage("panel.jpg", "GlassBreak")
        r = analyze_rgb_damage(img_array,   "HardShading")
        print(r.detected, r.damage_percentage, r.confidence)
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"Cannot read image: {image_input}")
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise TypeError("image_input must be str path or np.ndarray")

    if class_name not in CLASS_REGISTRY:
        raise ValueError(
            f"Unknown class '{class_name}'. "
            f"Available: {list(CLASS_REGISTRY.keys())}"
        )

    return CLASS_REGISTRY[class_name](image)


# ══════════════════════════════════════════════════════════════════
#  CLASS 4: SOFT SHADING / SOILING
#  Visual patterns:
#    • Brown dust patches  (img1) → S medium, H brownish
#    • Gray uniform soiling (img2) → S very low, full panel
#    • Bottom-heavy dust   (img3) → gravity settled, bot > top
#  Detection:
#    Gray signal (S<60) + Brown signal (H 8-38) + not-blue channel
#    → panel grid lines excluded via background subtraction
#    → spatial distribution → pattern classification
# ══════════════════════════════════════════════════════════════════

