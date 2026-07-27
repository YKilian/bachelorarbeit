import cv2
import numpy as np
import ENUM

# =========================================================
# KONFIGURATION & SCHWELLENWERTE
# =========================================================

DEBUG = False
IMAGE_PATH = "data/img/current.jpg"

# Mindestfläche in Pixeln
MIN_AREA = 300

# Maximale zulässige Höhe der Farb-Box im Fach (relativ zur Gesamthöhe des Fachs ROI).
# Bsp.: 0.55 bedeutet: Wenn die Farbkontur mehr als 55% der Gesamthöhe des Fachs
# einnimmt, gilt das Werkstück als verkantet.
MAX_HEIGHT_RATIO = 0.4

# Alternativ kannst du auch einen festen Pixelwert nutzen (z.B. MAX_HEIGHT_PX = 80)
# USE_PIXEL_HEIGHT = False

# Morphologisches Stapelelement
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# HSV-Farbräume
COLOR_RANGES = {
    "BLUE": [
        (np.array([100, 110, 120]), np.array([130, 255, 255]))
    ],
    "RED": [
        (np.array([0, 110, 120]), np.array([8, 255, 255])),
        (np.array([165, 110, 120]), np.array([179, 255, 255]))
    ],
    "WHITE": [
        (np.array([0, 0, 210]), np.array([180, 40, 255]))
    ]
}


# =========================================================
# HILFSFUNKTIONEN
# =========================================================

def get_clean_color_mask(hsv, color_name):
    """Erstellt eine bereinigte Binärmaske für die angeforderte Farbe."""
    ranges = COLOR_RANGES[color_name]
    full_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        mask = cv2.inRange(hsv, lower, upper)
        full_mask = cv2.bitwise_or(full_mask, mask)

    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)

    return full_mask


def analyze_container(roi, container_id, debug_canvas=None, global_offset=(0, 0)):
    if roi is None or roi.size == 0:
        return {"farbe": "", "lage": "ok", "flaeche": 0, "box_hoehe": 0}, None

    roi_h, roi_w = roi.shape[:2]

    # -----------------------------------------------------
    # 1. FARBERKENNUNG (EXAKT WIE GEHABT IM INNER_ROI)
    # -----------------------------------------------------
    margin_x = int(roi_w * 0.15)
    margin_y_top = int(roi_h * 0.38)
    margin_y_bottom = int(roi_h * 0.42)

    y1, y2 = margin_y_top, max(margin_y_top + 1, roi_h - margin_y_bottom)
    x1, x2 = margin_x, max(margin_x + 1, roi_w - margin_x)
    inner_roi = roi[y1:y2, x1:x2]

    blurred_inner = cv2.GaussianBlur(inner_roi, (5, 5), 0)
    hsv_inner = cv2.cvtColor(blurred_inner, cv2.COLOR_BGR2HSV)

    best_color = ""
    best_area = 0.0
    color_masks = {}

    for color_name in ["BLUE", "RED", "WHITE"]:
        mask = get_clean_color_mask(hsv_inner, color_name)
        color_masks[color_name] = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area and area >= MIN_AREA:
                best_area = area
                best_color = color_name

    # Fach gilt als leer, wenn im inneren Bereich keine Farbe liegt
    if not best_color or best_area < MIN_AREA:
        return {"farbe": "", "lage": "ok", "flaeche": 0, "box_hoehe": 0}, color_masks

    # -----------------------------------------------------
    # 2. HÖHENMESSUNG DER BOX IM GESAMTEN FACH-ROI
    # NUR FÜR DIE ERKANNTEN BEST_COLOR
    # -----------------------------------------------------
    blurred_full = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv_full = cv2.cvtColor(blurred_full, cv2.COLOR_BGR2HSV)

    # Erzeuge Maske im GANZEN Fach, aber NUR für die gefundene Farbe
    full_mask = get_clean_color_mask(hsv_full, best_color)
    contours_full, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_h = 0
    is_jammed = False
    full_box_coords = None  # (x, y, w, h) im ROI

    if contours_full and best_color != "":
        # Nimm die größte zusammenhängende Farbfläche im gesamten Fach
        largest_cnt = max(contours_full, key=cv2.contourArea)

        if cv2.contourArea(largest_cnt) >= MIN_AREA:
            # Aufrechte Bounding Box berechnen
            bx, by, bw, bh = cv2.boundingRect(largest_cnt)
            box_h = bh
            full_box_coords = (bx, by, bw, bh)

            # Relatives Verhältnis der Box-Höhe zur Fach-Höhe berechnen
            height_ratio = box_h / float(roi_h)

            # Prüfen, ob die Höhe den Schwellenwert überschreitet
            if height_ratio > MAX_HEIGHT_RATIO:
                is_jammed = True

    lage = "verkantet" if is_jammed else "ok"

    # -----------------------------------------------------
    # DEBUG EINZEICHNUNG
    # -----------------------------------------------------
    if DEBUG and debug_canvas is not None:
        gx, gy = global_offset

        # 1. Kleines inneres Messfenster für Farberkennung (Gelb)
        cv2.rectangle(debug_canvas, (gx + x1, gy + y1), (gx + x2, gy + y2), (0, 255, 255), 1)

        # 2. Bounding Box um die Farbstruktur im vollen ROI
        if full_box_coords is not None:
            bx, by, bw, bh = full_box_coords
            rect_color = (0, 0, 255) if is_jammed else (255, 200, 0)  # Rot bei Verkantung, sonst hellblau
            cv2.rectangle(debug_canvas, (gx + bx, gy + by), (gx + bx + bw, gy + by + bh), rect_color, 2)

    return {
        "farbe": best_color,
        "lage": lage,
        "flaeche": int(best_area),
        "box_hoehe": box_h
    }, color_masks


# =========================================================
# HAUPTPROGRAMM
# =========================================================

def main(soll_zustand):
    fokusbereiche = ENUM.FOKUSBEREICHE
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print(f"Fehler: Bild '{IMAGE_PATH}' konnte nicht geladen werden.")
        return []

    debug_canvas = image.copy() if DEBUG else None

    combined_masks = {c: np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8) for c in
                      ["BLUE", "RED", "WHITE"]} if DEBUG else {}

    statistics = []

    for idx, (y_start, y_ende, x_start, x_ende) in enumerate(fokusbereiche):
        roi = image[y_start:y_ende, x_start:x_ende]

        result, masks = analyze_container(roi, idx, debug_canvas, global_offset=(x_start, y_start))

        farbe = result["farbe"]
        lage = result["lage"]
        box_hoehe = result["box_hoehe"]

        anomalien = []

        soll_farbe = soll_zustand[idx]["Belegung"]
        if farbe != soll_farbe:
            anomalien.append("Farbe")

        if lage == "verkantet":
            anomalien.append("Verkantung")

        statistics.append({
            "Belegung": farbe,
            "Anomalien": anomalien
        })

        if DEBUG and debug_canvas is not None:
            # Rahmen um das gesamte Fach (Grün = OK, Rot = Anomalie)
            box_color = (0, 255, 0) if len(anomalien) == 0 else (0, 0, 255)
            cv2.rectangle(debug_canvas, (x_start, y_start), (x_ende, y_ende), box_color, 2)

            label = f"F{idx}: {farbe if farbe else 'EMPTY'} (Soll:{soll_farbe})"
            sub_label = f"Hoehe: {box_hoehe}px"

            cv2.putText(debug_canvas, label, (x_start + 5, y_start + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)
            cv2.putText(debug_canvas, sub_label, (x_start + 5, y_start + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (200, 200, 200), 1)

            if len(anomalien) > 0:
                cv2.putText(debug_canvas, f"! {','.join(anomalien)} !", (x_start + 5, y_ende - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

            # Masken zusammenfügen
            if masks:
                mx = int((x_ende - x_start) * 0.15)
                my_top = int((y_ende - y_start) * 0.38)

                for c_name, mask_patch in masks.items():
                    combined_masks[c_name][y_start + my_top:y_start + my_top + mask_patch.shape[0],
                    x_start + mx:x_start + mx + mask_patch.shape[1]] = mask_patch

    if DEBUG:
        cv2.imwrite("debug_overview.jpg", debug_canvas)
        for c_name, mask_img in combined_masks.items():
            cv2.imwrite(f"debug_mask_{c_name}.jpg", mask_img)

        cv2.imshow("Debug Overview", debug_canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return statistics