import random
from unittest import case

import cv2
import numpy as np
import os

import ENUM


def generiere_sps_state(sps_daten, generiere_fehler=False):
    """
    Generiert ein zusammengebautes Testbild des Hochregallagers basierend auf den
    SPS-Lagerdaten. Liest den aktuellen Belegungszustand aus, schneidet die Fächer
    aus den entsprechenden Basisbildern zu und fügt sie in ein leeres Hintergrundbild ein.
    """
    # --- 1. INITIALISIERUNG & HINTERGRUND-VORBEREITUNG ---
    scale_factor = 0.3  # Skalierungsfaktor zur Reduzierung der Bildgröße für performante Verarbeitung
    datenbank_pfad = "data/img/"  # Basisverzeichnis für das Bildmaterial

    # Basishintergrund (Regal mit leeren Containern) laden
    original_hintergrund = cv2.imread(os.path.join(datenbank_pfad, "empty_container.jpg"))

    # Für generierung von Fehlbildern, Mapping mit Gewichtung der Eintrittswahrscheinlichkeit des jeweiligen Fehlers
    gewichtung_anomalien = ENUM.GEWICHTUNG_ANOMALIEN

    if original_hintergrund is None:
        print("Fehler: 'empty_container.jpg' nicht gefunden!")
        return None

    # Für generierung von Fehlbildern, Mapping mit Gewichtung der Eintrittswahrscheinlichkeit des jeweiligen Fehlers
    gewichtung_anomalien = ENUM.GEWICHTUNG_ANOMALIEN

    # Ziel-Dimensionen berechnen und Hintergrund skalieren
    original_breite = int(original_hintergrund.shape[1] * scale_factor)
    original_hoehe = int(original_hintergrund.shape[0] * scale_factor)
    hintergrund = cv2.resize(original_hintergrund, (original_breite, original_hoehe), interpolation=cv2.INTER_AREA)

    # Fokusbereiche der einzelnen Regalfächer aus den Enums laden
    fokusbereiche = ENUM.FOKUSBEREICHE

    # SPS-Bestandsdaten extrahieren
    stock_items = sps_daten["payload"]["stockItems"]

    # Debug-Ausgabe aller empfangenen SPS-Einträge
    for item in stock_items:
        print(item)

    # Array zum gegenprüfen in der finalen auswertung. Enthält die Zustände, die das Kamerasystem tatsächlich hätte sehen sollen
    tatsaechlicher_zustand = []

    # --- 2. VERARBEITUNG DER EINZELNEN REGALFÄCHER ---
    fehlerhaftes_fach = random.randint(0, len(stock_items) - 1)
    for idx, fach in enumerate(stock_items):
        reihe = fach['location'][0]  # Zeilenbezeichnung (z. B. 'A', 'B', 'C')
        soll_farbe = fach['workpiece']['type']  # Werkstückfarbe

        if generiere_fehler and fehlerhaftes_fach == idx:
            # Für generierung von Fehlbildern, Mapping mit Gewichtung der Eintrittswahrscheinlichkeit des jeweiligen Fehlers
            gewichtung_anomalien = ENUM.GEWICHTUNG_ANOMALIEN_WERKSTUEK if soll_farbe != "" else ENUM.GEWICHTUNG_ANOMALIEN_OHNE_WERKSTUEK
            erzeugte_anomalie = str(np.random.choice(ENUM.ANOMALIEN, p=gewichtung_anomalien))
            neuer_zustand = ""
            match erzeugte_anomalie:
                case "Farbe":
                    kopie_basisbilder = ENUM.BASISBILDER[reihe].copy()
                    kopie_basisbilder[""] = "empty_container.jpg"
                    kopie_basisbilder.pop(soll_farbe)
                    neuer_zustand = random.choice(list(kopie_basisbilder.items()))
                    tatsaechlicher_zustand.append({"Belegung": neuer_zustand[0], "Anomalien": [erzeugte_anomalie]})
                case "Verkantung":
                    quellbild_normalzustand = ENUM.BASISBILDER[reihe][soll_farbe]
                    quellbild_verkantung = quellbild_normalzustand.replace("ok", "error")
                    neuer_zustand = [soll_farbe, quellbild_verkantung]
                    tatsaechlicher_zustand.append({"Belegung": neuer_zustand[0], "Anomalien": [erzeugte_anomalie]})
                case "Behälter_Rotiert":
                    neuer_zustand = ["", "rotated_container.jpg"]
                    tatsaechlicher_zustand.append({"Belegung": neuer_zustand[0], "Anomalien": [erzeugte_anomalie, "Farbe"]})
                case "Behälter_Fehlt":
                    neuer_zustand = ["", "no_container.jpg"]
                    tatsaechlicher_zustand.append({"Belegung": neuer_zustand[0], "Anomalien": [erzeugte_anomalie]})

            quellbild = neuer_zustand[1]
            # Quellbild laden, falls ein Dateipfad-String übergeben wurde
            if isinstance(quellbild, str):
                quellbild = cv2.imread(os.path.join(datenbank_pfad, quellbild))
            else:
                quellbild = quellbild

            # Validierung, ob das Bild geladen werden konnte
            if quellbild is None:
                print(f"Fehler: Bild für Reihe {reihe}, Farbe {soll_farbe} konnte nicht geladen werden!")
                continue

            # Quellbild auf die globale Zielgröße skalieren
            quellbild = cv2.resize(quellbild, (original_breite, original_hoehe), interpolation=cv2.INTER_AREA)
            print(f"Fach {idx} ({fach['location']}): {soll_farbe}")

            # Koordinaten des Ziel-Fachs laden (Format: y_start, y_ende, x_start, x_ende)
            y_start, y_ende, x_start, x_ende = fokusbereiche[idx]
            hoehe, breite = y_ende - y_start, x_ende - x_start

            # Relevantes Fach aus dem Quellbild ausschneiden und anpassen
            quellbild_ausschnitt = quellbild[y_start:y_ende, x_start:x_ende].copy()
            quellbild_ausschnitt_skaliert = cv2.resize(quellbild_ausschnitt, (breite, hoehe))

            # Ausgeschnittenes Fach in das Hintergrundbild einbetten
            hintergrund[y_start:y_ende, x_start:x_ende] = quellbild_ausschnitt_skaliert

        else:
            # Prüfen, ob das Fach mit einem Werkstück belegt ist
            tatsaechlicher_zustand.append({"Belegung": fach['workpiece']['type'], "Anomalien": []})
            if fach["workpiece"]["type"] != "":
                # Basisbild-Referenz aus der Mapping-Tabelle ermitteln
                quellbild = ENUM.BASISBILDER[reihe][soll_farbe]

                # Quellbild laden, falls ein Dateipfad-String übergeben wurde
                if isinstance(quellbild, str):
                    quellbild = cv2.imread(os.path.join(datenbank_pfad, quellbild))
                else:
                    quellbild = quellbild

                # Validierung, ob das Bild geladen werden konnte
                if quellbild is None:
                    print(f"Fehler: Bild für Reihe {reihe}, Farbe {soll_farbe} konnte nicht geladen werden!")
                    continue

                # Quellbild auf die globale Zielgröße skalieren
                quellbild = cv2.resize(quellbild, (original_breite, original_hoehe), interpolation=cv2.INTER_AREA)
                print(f"Fach {idx} ({fach['location']}): {soll_farbe}")

                # Koordinaten des Ziel-Fachs laden (Format: y_start, y_ende, x_start, x_ende)
                y_start, y_ende, x_start, x_ende = fokusbereiche[idx]
                hoehe, breite = y_ende - y_start, x_ende - x_start

                # Relevantes Fach aus dem Quellbild ausschneiden und anpassen
                quellbild_ausschnitt = quellbild[y_start:y_ende, x_start:x_ende].copy()
                quellbild_ausschnitt_skaliert = cv2.resize(quellbild_ausschnitt, (breite, hoehe))

                # Ausgeschnittenes Fach in das Hintergrundbild einbetten
                hintergrund[y_start:y_ende, x_start:x_ende] = quellbild_ausschnitt_skaliert

    # --- 3. SPEICHERUNG & AUSGABE ---
    finales_bild = hintergrund
    os.makedirs(os.path.dirname("data/img/"), exist_ok=True)
    cv2.imwrite(ENUM.IMAGE_PATH, hintergrund)
    print(f"-> Neues Testbild erfolgreich skaliert und als {ENUM.IMAGE_PATH} gespeichert.")

    return tatsaechlicher_zustand