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
GEWICHTUNG_ANOMALIEN = [0.6, 0.2, 0.1, 0.1]