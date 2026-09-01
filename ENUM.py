import cv2
import numpy as np

# === FOKUSBEREICHE ===
# Bildkoordinaten und Skalierung der jeweiligen Fächer im Bild
# (y_start, y_end, x_start, x_end)
FOKUSBEREICHE = [
        (150, 295, 210, 335), (150, 295, 515, 640), (150, 295, 820, 955),
        (340, 485, 210, 335), (340, 485, 515, 640), (340, 485, 820, 945),
        (530, 675, 210, 335), (530, 675, 515, 640), (530, 675, 820, 935)
    ]

# === ZIELBILDER ===
# Zuordnung von Farben und Reihen zu den jeweiligen Basisbildern
BASISBILDER = {
    "A": {
        "BLUE": "ok_bwr.jpg",
        "RED": "ok_rbw.jpg",
        "WHITE": "ok_wrb.jpg"
    },
    "B": {
        "BLUE": "ok_rbw.jpg",
        "RED": "ok_wrb.jpg",
        "WHITE": "ok_bwr.jpg"
    },
    "C": {
        "BLUE": "ok_wrb.jpg",
        "RED": "ok_bwr.jpg",
        "WHITE": "ok_rbw.jpg"
    }
}

# === FÄCHER ===
# Zuordnen von möglichen Belegungen, Anomalien und deren Eintrittswahrscheinlichkeit für jedes Fach
BELEGUNGEN = ["RED", "BLUE", "WHITE", ""]
ANOMALIEN = ["Farbe", "Verkantung", "Behälter_Rotiert", "Behälter_Fehlt"]
GEWICHTUNG_ANOMALIEN_WERKSTUEK = [0.6, 0.2, 0.1, 0.1]
GEWICHTUNG_ANOMALIEN_OHNE_WERKSTUEK = [0.6, 0.0, 0.2, 0.2]

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
    ],
    "BLACK": [
        (np.array([0, 0, 65]), np.array([180, 95, 140]))
    ]
}