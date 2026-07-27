import cv2
import numpy as np
import os


def generiere_test_zustand(layout, datenbank_pfad="data/img/"):
    scale_factor = 0.3  # Exakt derselbe Faktor wie in deiner Pipeline!

    # 1. Basis-Hintergrund laden und direkt SKALIEREN
    original_hintergrund = cv2.imread(os.path.join(datenbank_pfad, "no_container.jpg"))
    if original_hintergrund is None:
        print("Fehler: 'no_container.jpg' nicht gefunden!")
        return None

    # Skalierung auf die Zielgröße, zu der deine ROIs passen
    width = int(original_hintergrund.shape[1] * scale_factor)
    height = int(original_hintergrund.shape[0] * scale_factor)
    hintergrund = cv2.resize(original_hintergrund, (width, height), interpolation=cv2.INTER_AREA)

    rois = [
        (150, 285, 210, 315), (150, 285, 535, 640), (150, 285, 850, 955),
        (340, 475, 210, 315), (340, 475, 535, 640), (340, 475, 840, 945),
        (530, 665, 210, 315), (530, 665, 535, 640), (530, 665, 830, 935)
    ]

    gesamt_bilder_namen = {
        "leer": "empty_container.jpg",
        "rbw_ok": "ok_rbw.jpg",
        "rbw_error": "error_rbw.jpg",
        "bwr_ok": "ok_bwr.jpg",
        "bwr_error": "error_bwr.jpg",
        "wrb_ok": "ok_wrb.jpg",
        "wrb_error": "error_wrb.jpg",
    }

    # Gesamtbilder laden und ebenfalls direkt SKALIEREN
    gesamt_bilder = {}
    for name, datei in gesamt_bilder_namen.items():
        img = cv2.imread(os.path.join(datenbank_pfad, datei))
        if img is not None:
            # Jedes Quellbild auf die gleiche Größe bringen wie den Hintergrund
            img_scaled = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            gesamt_bilder[name] = img_scaled

    gewuenschte_zustaende = np.array(layout).flatten()

    # 3. Das Bild Fach für Fach zusammenbauen
    for idx, zustand_name in enumerate(gewuenschte_zustaende):
        y_start, y_end, x_start, x_end = rois[idx]
        h, w = y_end - y_start, x_end - x_start

        quell_bild = gesamt_bilder.get(zustand_name)
        if quell_bild is None:
            continue

        # Da quell_bild jetzt herunterskaliert ist, passen die Koordinaten exakt!
        snippet = quell_bild[y_start:y_end, x_start:x_end].copy()
        snippet_resized = cv2.resize(snippet, (w, h))

        # In das Hintergrundbild einsetzen
        hintergrund[y_start:y_end, x_start:x_end] = snippet_resized

    # Ergebnis als current.jpg speichern
    os.makedirs(os.path.dirname("data/img/"), exist_ok=True)
    cv2.imwrite("data/img/current.jpg", hintergrund)
    print("-> Neues Testbild erfolgreich skaliert und als 'data/img/current.jpg' gespeichert.")
    return hintergrund