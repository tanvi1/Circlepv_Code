"""
BacksheetDamage detector  (v2 - clean / single-detection-per-class)
---------------------------------------------------------------------
Classical-CV detector for exactly the two defect types you asked for:
 
    1. Crack / Tear   - a torn/ripped patch of backsheet film. It shows up
                         as an irregular, jagged blob whose pixels swing
                         between very bright (exposed under-layer / flap)
                         and very dark (the hole/void behind it), i.e. it
                         has HIGH internal brightness variation (std).
 
    2. Ballooning      - a smooth dome-shaped bulge from delamination /
                         trapped gas. It shows up as a soft, ROUND blob
                         with a gentle brightness gradient (no jaggedness),
                         i.e. HIGH circularity + low internal noise.
 
Why the previous version looked "useless"
-------------------------------------------
A single generic "is this pixel smooth?" mask fires on a lot of normal,
undamaged backsheet (any flat cell between grid/ridge lines is also
"smooth"), so it fragmented into a dozen boxes scattered over the whole
panel. This version fixes that by:
 
  * Running TWO independent, purpose-built detectors (one per defect
    type) instead of one generic texture mask.
  * Each detector returns AT MOST ONE region -- its single best-scoring
    candidate -- instead of every connected component above a pixel
    threshold.
  * Each detector has an acceptance gate (a minimum score) so that if
    that defect type genuinely isn't present, it reports nothing rather
    than forcing a guess.
 
This is still a classical, untrained heuristic (no model weights), tuned
against the two sample images you provided. It will need re-tuning
(the numbers in __init__) if you point it at a very different panel
style / lighting setup -- a trained segmentation model would generalize
better, but this gives you a working, inspectable baseline right now.
 
Usage
-----
    from backsheet_damage_detector import BacksheetDamage
 
    detector = BacksheetDamage()
    annotated, damage_pct, detections = detector.detect(image_bgr)
 
`image_bgr` is a numpy array (e.g. from cv2.imread, a video frame, or a
decoded upload) -- the "input image array" you asked for.
"""
 
import cv2
import numpy as np
 
 
class BacksheetDamage:
    class_name = "BacksheetDamage"
 
    def __init__(
        self,
        # ---- shared ----
        border_ignore_frac=0.08,     # ignore this much of the outer frame
                                      # (avoids window frames / mounting
                                      # hardware / curtains at the edges)
        min_area_frac=0.01,          # ignore blobs smaller than 1% of image
        max_area_frac=0.35,          # ignore blobs bigger than 35% of image
                                      # (almost certainly background, not
                                      # a localized defect)
        # ---- crack / tear detector ----
        crack_local_win=9,
        crack_smooth_percentile=35,
        crack_min_area_frac=0.005,
        crack_min_internal_std=12.0, # acceptance gate: needs this much
                                      # brightness contrast inside the blob
        # ---- ballooning detector ----
        balloon_small_sigma_frac=0.035,
        balloon_large_sigma_frac=0.17,
        balloon_min_area_frac=0.02,
        balloon_min_circularity=0.68,  # acceptance gate: needs to be this round
    ):
        self.border_ignore_frac = border_ignore_frac
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
 
        self.crack_local_win = crack_local_win
        self.crack_smooth_percentile = crack_smooth_percentile
        self.crack_min_area_frac = crack_min_area_frac
        self.crack_min_internal_std = crack_min_internal_std
 
        self.balloon_small_sigma_frac = balloon_small_sigma_frac
        self.balloon_large_sigma_frac = balloon_large_sigma_frac
        self.balloon_min_area_frac = balloon_min_area_frac
        self.balloon_min_circularity = balloon_min_circularity
 
    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _touches_border(bbox, shape, tol=3):
        x, y, w, h = bbox
        H, W = shape
        sides = 0
        if x <= tol: sides += 1
        if y <= tol: sides += 1
        if x + w >= W - tol: sides += 1
        if y + h >= H - tol: sides += 1
        return sides
 
    def _border_mask(self, h, w):
        m = int(self.border_ignore_frac * min(h, w))
        mask = np.zeros((h, w), np.uint8)
        mask[m:h - m, m:w - m] = 255
        return mask
 
    # ---- crack / tear: find the single most "torn-looking" blob ---- #
    def _find_crack(self, gray):
        h, w = gray.shape
        total = float(h * w)
 
        g = gray.astype(np.float32)
        mean = cv2.boxFilter(g, -1, (self.crack_local_win, self.crack_local_win))
        sq = cv2.boxFilter(g * g, -1, (self.crack_local_win, self.crack_local_win))
        var = np.clip(sq - mean * mean, 0, None)
        var_n = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
 
        thresh_val = np.percentile(var_n, self.crack_smooth_percentile)
        mask = (var_n <= thresh_val).astype(np.uint8) * 255
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        best = None  # (std, bbox, area_pct, contour)
        for c in contours:
            area = cv2.contourArea(c)
            frac = area / total
            if frac < max(self.min_area_frac, self.crack_min_area_frac) or frac > self.max_area_frac:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if self._touches_border((x, y, bw, bh), (h, w)) >= 2:
                continue
            roi = gray[y:y + bh, x:x + bw]
            std = float(roi.std())
            if best is None or std > best[0]:
                best = (std, (x, y, bw, bh), frac * 100.0, c)
 
        if best is not None and best[0] >= self.crack_min_internal_std:
            return {"label": "Crack/Tear", "bbox": best[1], "area_pct": round(best[2], 2),
                    "contour": best[3], "score": round(best[0], 2)}
        return None
 
    # ---- ballooning: find the single most "smooth round bulge" blob ---- #
    def _find_balloon(self, gray):
        h, w = gray.shape
        total = float(h * w)
 
        g = gray.astype(np.float32)
        s_small = max(3.0, self.balloon_small_sigma_frac * min(h, w))
        s_large = max(8.0, self.balloon_large_sigma_frac * min(h, w))
        dog = cv2.GaussianBlur(g, (0, 0), s_small) - cv2.GaussianBlur(g, (0, 0), s_large)
 
        dog_pos = np.clip(dog, 0, None)
        dog_pos[self._border_mask(h, w) == 0] = 0
        dog_n = cv2.normalize(dog_pos, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(dog_n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        best = None  # (circularity, bbox, area_pct, contour)
        for c in contours:
            area = cv2.contourArea(c)
            frac = area / total
            if frac < max(self.min_area_frac, self.balloon_min_area_frac) or frac > self.max_area_frac:
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circ = float(4 * np.pi * area / (perim ** 2))
            x, y, bw, bh = cv2.boundingRect(c)
            if self._touches_border((x, y, bw, bh), (h, w)) >= 2:
                continue
            if best is None or circ > best[0]:
                best = (circ, (x, y, bw, bh), frac * 100.0, c)
 
        if best is not None and best[0] >= self.balloon_min_circularity:
            return {"label": "Ballooning", "bbox": best[1], "area_pct": round(best[2], 2),
                    "contour": best[3], "score": round(best[0], 2)}
        return None
 
    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def detect(self, image_bgr):
        """
        Parameters
        ----------
        image_bgr : np.ndarray  (H, W, 3) BGR image array
 
        Returns
        -------
        annotated_image   : np.ndarray -- copy of input with box/label drawn
        damage_percentage : float      -- summed defect area as % of image
        detections         : list[dict] -- 0, 1, or 2 entries (one per type)
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image array passed to BacksheetDamage.detect()")
 
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
 
        crack = self._find_crack(gray_eq)
        balloon = self._find_balloon(gray_eq)
 
        detections = [d for d in (crack, balloon) if d is not None]
 
        annotated = image_bgr.copy()
        colors = {"Crack/Tear": (0, 0, 255), "Ballooning": (0, 165, 255)}
 
        for d in detections:
            x, y, bw, bh = d["bbox"]
            color = colors[d["label"]]
            cv2.drawContours(annotated, [d["contour"]], -1, color, 2)
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 1)
            tag = f"{self.class_name}: {d['label']} {d['area_pct']:.1f}%"
            ty = y - 8 if y - 8 > 10 else y + bh + 18
            cv2.putText(annotated, tag, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
            d.pop("contour")  # not JSON/print friendly, drop from the returned dict
 
        damage_percentage = round(sum(d["area_pct"] for d in detections), 2)
 
        summary = f"{self.class_name} total damage: {damage_percentage:.2f}%"
        cv2.rectangle(annotated, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(annotated, summary, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
 
        return annotated, damage_percentage, detections
 

# ---------------------------------------------------------------------- #
# demo / CLI
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    import glob

    detector = BacksheetDamage()
    paths = sys.argv[1:] if len(sys.argv) > 1 else glob.glob("*.png")

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[skip] could not read {p}")
            continue
        annotated, pct, dets = detector.detect(img)
        out_path = p.rsplit(".", 1)[0] + "_annotated.png"
        cv2.imwrite(out_path, annotated)
        print(f"{p} -> damage {pct:.2f}% | detections: {dets} | saved: {out_path}")