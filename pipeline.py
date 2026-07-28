import cv2
import numpy as np
import ENUM

# =========================================================
# KONFIGURATION & SCHWELLENWERTE
# =========================================================

DEBUG = True
IMAGE_PATH = "data/img/current.jpg"

# Mindestfläche in Pixeln für Farbobjekte (Inhalt)
MIN_AREA = 300

# Maximale zulässige Höhe der Farb-Box im Fach (relativ zur Gesamthöhe des Fachs ROI).
MAX_HEIGHT_RATIO = 0.4

# Schwellenwerte für den schwarzen Behälter
MIN_CONTAINER_AREA = 3700
MIN_CONTAINER_HEIGHT_PX = 50  # Ein echter Behälter ist z.B. mindestens 50px hoch

# Schwellenwert für die Aussparung an der Unterseite des Behälters:
MIN_GAP_RATIO = 0.06 # Mindestens 15% der Breite müssen im unteren Bereich "frei" sein

# Morphologisches Stapelelement
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# HSV-Farbräume für den Inhalt
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

def get_black_container_mask(hsv_roi):
    """
    Erstellt eine Binärmaske für dunkle/schwarze Objekte (Behälter).
    """
    lower_black = np.array([0, 0, 65])
    upper_black = np.array([180, 95, 140])

    # lower_black = np.array([0, 0, 7])
    # upper_black = np.array([180, 102, 140])

    mask = cv2.inRange(hsv_roi, lower_black, upper_black)

    # Rauschunterdrückung & Lücken im Plastik schließen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def analyze_container_presence_and_orientation(roi, debug_canvas=None, global_offset=(0, 0)):
    """
    Prüft, ob ein schwarzer Behälter existiert und ob seine Boden-Aussparung
    (Füße) sichtbar ist (= korrekte Ausrichtung).
    """
    if roi is None or roi.size == 0:
        return {"vorhanden": False, "ausrichtung": "unbekannt", "gap_ratio": 0.0, "flaeche": 0, "hoehe": 0}

    roi_h, roi_w = roi.shape[:2]

    # 1. Bild aufbereiten & Schwarze Maske erstellen
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = get_black_container_mask(hsv)

    # Contours des schwarzen Objekts finden
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"vorhanden": False, "ausrichtung": "fehlt", "gap_ratio": 0.0, "flaeche": 0, "hoehe": 0}

    # Größtes schwarzes Objekt im Fach suchen
    largest_cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_cnt)
    bx, by, bw, bh = cv2.boundingRect(largest_cnt)

    # --- PRÜFUNG 1: PRESENCE (Vorhandensein) ---
    if area < MIN_CONTAINER_AREA or bh < MIN_CONTAINER_HEIGHT_PX:
        return {
            "vorhanden": False,
            "ausrichtung": "fehlt",
            "gap_ratio": 0.0,
            "flaeche": int(area),
            "hoehe": bh
        }

    # --- PRÜFUNG 2: ORIENTATION (Aussparung an der Unterseite) ---
    # Wir betrachten das unterste Viertel (25%) der Bounding Box des Behälters
    bottom_zone_y_start = by + int(bh * 0.75)
    bottom_zone_h = bh - int(bh * 0.75)

    # Ausschnitt der Maske im unteren Bereich des Behälters
    bottom_mask = mask[bottom_zone_y_start: by + bh, bx: bx + bw]

    if bottom_mask.size == 0:
        return {
            "vorhanden": True,
            "ausrichtung": "verdreht",
            "gap_ratio": 0.0,
            "flaeche": int(area),
            "hoehe": bh
        }

    # Wir zählen spaltenweise (vertikal), wie viele schwarze Pixel im Bodenbereich liegen.
    column_pixel_counts = np.sum(bottom_mask > 0, axis=0)

    # Eine Spalte gilt als "Aussparung/Lücke", wenn weniger als 30% ihrer Höhe schwarz sind
    empty_columns = np.sum(column_pixel_counts < (bottom_zone_h * 0.3))

    # Relativer Anteil der Lücke bezogen auf die Gesamtbreite des Behälters
    gap_ratio = empty_columns / float(bw)

    # Entscheidung: Ist die Aussparung breit genug?
    is_correctly_oriented = gap_ratio >= MIN_GAP_RATIO
    ausrichtung = "ok" if is_correctly_oriented else "verdreht"

    # -----------------------------------------------------
    # DEBUG EINZEICHNUNG BEHÄLTER
    # -----------------------------------------------------
    if DEBUG and debug_canvas is not None:
        gx, gy = global_offset

        # Behälter-Bounding-Box (Blau)
        cv2.rectangle(debug_canvas, (gx + bx, gy + by), (gx + bx + bw, gy + by + bh), (255, 100, 0), 2)

        # Untere Messzone für die Aussparung (Grün bei OK, Rot bei Fehler)
        zone_color = (0, 255, 0) if is_correctly_oriented else (0, 0, 255)
        cv2.rectangle(debug_canvas,
                      (gx + bx, gy + bottom_zone_y_start),
                      (gx + bx + bw, gy + by + bh),
                      zone_color, 1)

    return {
        "vorhanden": True,
        "ausrichtung": ausrichtung,
        "gap_ratio": round(gap_ratio, 2),
        "flaeche": int(area),
        "hoehe": bh
    }


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
    """
    Analysiert den Inhalt des Behälters (Farbe, Fläche, Verkantung/Höhe).
    """
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

    # Inhalt gilt als leer, wenn im inneren Bereich keine Farbe liegt
    if not best_color or best_area < MIN_AREA:
        return {"farbe": "", "lage": "ok", "flaeche": 0, "box_hoehe": 0}, color_masks

    # -----------------------------------------------------
    # 2. HÖHENMESSUNG DER BOX IM GESAMTEN FACH-ROI
    # NUR FÜR DIE ERKANNTEN BEST_COLOR
    # -----------------------------------------------------
    blurred_full = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv_full = cv2.cvtColor(blurred_full, cv2.COLOR_BGR2HSV)

    full_mask = get_clean_color_mask(hsv_full, best_color)
    contours_full, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_h = 0
    is_jammed = False
    full_box_coords = None

    if contours_full and best_color != "":
        largest_cnt = max(contours_full, key=cv2.contourArea)

        if cv2.contourArea(largest_cnt) >= MIN_AREA:
            bx, by, bw, bh = cv2.boundingRect(largest_cnt)
            box_h = bh
            full_box_coords = (bx, by, bw, bh)

            height_ratio = box_h / float(roi_h)

            if height_ratio > MAX_HEIGHT_RATIO:
                is_jammed = True

    lage = "verkantet" if is_jammed else "ok"

    # -----------------------------------------------------
    # DEBUG EINZEICHNUNG INHALT
    # -----------------------------------------------------
    if DEBUG and debug_canvas is not None:
        gx, gy = global_offset

        # 1. Kleines inneres Messfenster für Farberkennung (Gelb)
        cv2.rectangle(debug_canvas, (gx + x1, gy + y1), (gx + x2, gy + y2), (0, 255, 255), 1)

        # 2. Bounding Box um die Farbstruktur im vollen ROI
        if full_box_coords is not None:
            bx, by, bw, bh = full_box_coords
            rect_color = (0, 0, 255) if is_jammed else (255, 200, 0)
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

        # 1. Analyse des Behälters (Präsenz & Ausrichtung)
        container_info = analyze_container_presence_and_orientation(
            roi, debug_canvas, global_offset=(x_start, y_start)
        )

        # 2. Analyse des Inhalts (Farbe, Fläche, Verkantung)
        content_info, masks = analyze_container(
            roi, idx, debug_canvas, global_offset=(x_start, y_start)
        )

        farbe = content_info["farbe"]
        lage = content_info["lage"]
        box_hoehe = content_info["box_hoehe"]

        anomalien = []

        # --- ANOMALIEN-PRÜFUNG ---
        # A) Behälter-Fehler
        if not container_info["vorhanden"]:
            anomalien.append("Behaelter fehlt")
        elif container_info["ausrichtung"] == "verdreht":
            anomalien.append("Behaelter verdreht")

        # B) Inhalts-Fehler
        soll_farbe = soll_zustand[idx]["Belegung"]
        if farbe != soll_farbe:
            anomalien.append("Farbe")

        if lage == "verkantet":
            anomalien.append("Verkantung")

        # --- STATISTIKEN ---
        statistics.append({
            "Belegung": farbe,
            "Behaelter_Vorhanden": container_info["vorhanden"],
            "Behaelter_Ausrichtung": container_info["ausrichtung"],
            "Behaelter_Flaeche": container_info["flaeche"],
            "Behaelter_Hoehe": container_info["hoehe"],
            "Anomalien": anomalien
        })

        # --- DEBUG OVERLAY ---
        if DEBUG and debug_canvas is not None:
            # Fach-Rahmen (Grün = OK, Rot = Anomalie)
            box_color = (0, 255, 0) if len(anomalien) == 0 else (0, 0, 255)
            cv2.rectangle(debug_canvas, (x_start, y_start), (x_ende, y_ende), box_color, 2)

            label = f"F{idx}: {farbe if farbe else 'EMPTY'} (Soll:{soll_farbe})"

            # NEU: Zeigt Fläche (A: ...px) und Höhe (H: ...px) des Behälters an
            sub_label = f"A:{container_info['flaeche']}px | H:{container_info['hoehe']}px | Gap:{int(container_info['gap_ratio'] * 100)}%"

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