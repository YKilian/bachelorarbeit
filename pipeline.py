import cv2
import numpy as np
from collections import defaultdict

# =========================================================
# KONFIGURATION
# =========================================================

DEBUG = False

IMAGE_PATH = "data/img/current.jpg"

SCALE_FACTOR = 1.0

# Mindestfläche, damit ein Werkstück als vorhanden gilt
MIN_AREA = 500

# Mindestfüllgrad für "korrekt eingelegt"
MIN_FILL_RATIO = 0.20

# Kernel für Morphologie
KERNEL = np.ones((5, 5), np.uint8)

# =========================================================
# HILFSFUNKTIONEN
# =========================================================

def create_color_masks(hsv):

    masks = {}

    # -----------------------------------------------------
    # ROT
    # -----------------------------------------------------

    red1 = cv2.inRange(
        hsv,
        np.array([0, 150, 50]),
        np.array([10, 255, 255])
    )

    red2 = cv2.inRange(
        hsv,
        np.array([170, 150, 50]),
        np.array([180, 255, 255])
    )

    # masks["rot"] = cv2.bitwise_or(red1, red2)
    masks["rot"] = cv2.inRange(
        hsv,
        np.array([0, 80, 50]),
        np.array([10, 255, 255])
    )

    # -----------------------------------------------------
    # BLAU
    # -----------------------------------------------------

    masks["blau"] = cv2.inRange(
        hsv,
        np.array([90, 80, 40]),
        np.array([130, 255, 255])
    )

    # -----------------------------------------------------
    # WEISS
    # -----------------------------------------------------

    masks["weiss"] = cv2.inRange(
        hsv,
        np.array([0, 0, 140]),
        np.array([180, 70, 255])
    )

    return masks


def clean_mask(mask):

    # Rauschen entfernen
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        KERNEL
    )

    # Löcher schließen
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        KERNEL
    )

    return mask


def analyze_container(roi, container_id):
    if roi is None or roi.size == 0:
        return {
            "farbe": "fehler", "lage": "ROI leer", "flaeche": 0,
            "kontur": None, "bounding_box": None, "fill_ratio": 0
        }

    # -----------------------------------------------------
    # ROI verkleinern
    # Nur relevanten Innenbereich verwenden
    # -----------------------------------------------------

    h, w = roi.shape[:2]

    margin_x = int(w * 0.15)
    margin_y_top = int(h * 0.10)
    margin_y_bottom = int(h * 0.35)

    inner_roi = roi[
        margin_y_top:h - margin_y_bottom,
        margin_x:w - margin_x
    ]

    # -----------------------------------------------------
    # Blur
    # -----------------------------------------------------

    blurred = cv2.GaussianBlur(inner_roi, (5, 5), 0)

    # -----------------------------------------------------
    # HSV
    # -----------------------------------------------------

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # -----------------------------------------------------
    # Farbmasken
    # -----------------------------------------------------

    masks = create_color_masks(hsv)

    best_color = "leer"
    best_area = 0
    best_contour = None
    best_mask = None

    # -----------------------------------------------------
    # Jede Farbe prüfen
    # -----------------------------------------------------

    for color_name, mask in masks.items():

        mask = clean_mask(mask)

        if DEBUG:
            cv2.imshow(
               f"Container {container_id} - {color_name}",
               mask
            )
            cv2.waitKey(0)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area > best_area:
                best_area = area
                best_color = color_name
                best_contour = contour
                best_mask = mask

    # -----------------------------------------------------
    # LEER?
    # -----------------------------------------------------

    if best_area < MIN_AREA:

        return {
            "farbe": "leer",
            "lage": "-",
            "flaeche": 0,
            "kontur": None,
            "bounding_box": None,
            "fill_ratio": 0
        }

    # -----------------------------------------------------
    # Bounding Box
    # -----------------------------------------------------

    x, y, w, h = cv2.boundingRect(best_contour)

    rect_area = w * h

    if rect_area == 0:
        fill_ratio = 0
    else:
        fill_ratio = best_area / rect_area

    # -----------------------------------------------------
    # Seitenverhältnis
    # -----------------------------------------------------

    aspect_ratio = w / h

    # -----------------------------------------------------
    # Verkantung erkennen
    # -----------------------------------------------------

    # Kriterien:
    # - schlechter Füllgrad
    # - stark verzogenes Seitenverhältnis

    if (
        fill_ratio < MIN_FILL_RATIO
        or aspect_ratio < 0.4
        or aspect_ratio > 6.0
    ):
        orientation = "verkantet"
    else:
        orientation = "ok"

    # -----------------------------------------------------
    # Kontur zurück auf ROI-Koordinaten verschieben
    # -----------------------------------------------------

    best_contour = best_contour.copy()

    best_contour[:, :, 0] += margin_x
    best_contour[:, :, 1] += margin_y_top

    bbox_global = (
        x + margin_x,
        y + margin_y_top,
        w,
        h
    )

    return {
        "farbe": best_color,
        "lage": orientation,
        "flaeche": int(best_area),
        "kontur": best_contour,
        "bounding_box": bbox_global,
        "fill_ratio": fill_ratio
    }

def main():
    # =========================================================
    # ROIs
    # Format:
    # (y_start, y_end, x_start, x_end)
    # =========================================================

    rois = [
        (180, 285, 210, 315),
        (180, 285, 535, 640),
        (180, 285, 850, 955),

        (370, 475, 210, 315),
        (370, 475, 535, 640),
        (370, 475, 840, 945),

        (560, 665, 210, 315),
        (560, 665, 535, 640),
        (560, 665, 830, 935)
    ]

    # =========================================================
    # BILD LADEN
    # =========================================================

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Fehler: Bild konnte nicht geladen werden.")
        exit()

    # Bild skalieren
    width = int(image.shape[1] * SCALE_FACTOR)
    height = int(image.shape[0] * SCALE_FACTOR)

    image = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    # Ergebnisbild
    result_image = image.copy()

    # =========================================================
    # STATISTIK
    # =========================================================

    statistics = []

    color_counts = defaultdict(int)

    # =========================================================
    # HAUPTSCHLEIFE
    # =========================================================

    for idx, (y1, y2, x1, x2) in enumerate(rois):

        roi = image[y1:y2, x1:x2]

        result = analyze_container(
            roi,
            idx + 1
        )

        color = result["farbe"]
        orientation = result["lage"]
        area = result["flaeche"]

        statistics.append({
            "Container": idx + 1,
            "Farbe": color,
            "Lage": orientation,
            "Fläche": area
        })

        if color != "leer":
            color_counts[color] += 1

        # -----------------------------------------------------
        # ROI zeichnen
        # -----------------------------------------------------

        cv2.rectangle(
            result_image,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2
        )

        # -----------------------------------------------------
        # Kontur zeichnen
        # -----------------------------------------------------

        if result["kontur"] is not None:

            contour_global = result["kontur"].copy()

            contour_global[:, :, 0] += x1
            contour_global[:, :, 1] += y1

            cv2.drawContours(
                result_image,
                [contour_global],
                -1,
                (0, 255, 0),
                2
            )

        # -----------------------------------------------------
        # Bounding Box zeichnen
        # -----------------------------------------------------

        if result["bounding_box"] is not None:

            bx, by, bw, bh = result["bounding_box"]

            cv2.rectangle(
                result_image,
                (x1 + bx, y1 + by),
                (x1 + bx + bw, y1 + by + bh),
                (0, 0, 255),
                2
            )

        # -----------------------------------------------------
        # Text
        # -----------------------------------------------------

        text = f"{color.upper()} | {orientation}"

        cv2.putText(
            result_image,
            text,
            (x1 + 5, y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            result_image,
            f"A:{area}",
            (x1 + 5, y1 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

    # =========================================================
    # AUSGABE
    # =========================================================

    print("\n===================================================")
    print("ERKENNUNGSERGEBNIS")
    print("===================================================\n")

    print(
        "{:<12} {:<10} {:<15} {:<10}".format(
            "Container",
            "Farbe",
            "Lage",
            "Fläche"
        )
    )

    print("-" * 55)

    for stat in statistics:

        print(
            "{:<12} {:<10} {:<15} {:<10}".format(
                stat["Container"],
                stat["Farbe"],
                stat["Lage"],
                stat["Fläche"]
            )
        )

    # print("\n===================================================")
    # print("ZUSAMMENFASSUNG")
    # print("===================================================\n")

    # for color, count in color_counts.items():
    #     print(f"{color.upper()}: {count}")

    # =========================================================
    # DEBUG / VISUALISIERUNG
    # =========================================================

    if DEBUG:

        cv2.imshow(
            "Ergebnis",
            result_image
        )

        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return statistics