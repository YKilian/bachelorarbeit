from unittest import case

import cv2
import numpy as np
import os

import ENUM


def generiere_sps_state(sps_daten, generate_error=False):
    """
    Generiert ein zusammengebautes Testbild des Hochregallagers basierend auf den
    SPS-Lagerdaten. Liest den aktuellen Belegungszustand aus, schneidet die Fächer
    aus den entsprechenden Basisbildern zu und fügt sie in ein leeres Hintergrundbild ein.
    """
    # --- 1. INITIALISIERUNG & HINTERGRUND-VORBEREITUNG ---
    scale_factor = 0.3  # Skalierungsfaktor zur Reduzierung der Bildgröße für performante Verarbeitung
    datenbank_pfad = "data/img/"  # Basisverzeichnis für das Bildmaterial

    # Basishintergrund (Leeres Regal ohne Container) laden
    original_hintergrund = cv2.imread(os.path.join(datenbank_pfad, "empty_container.jpg"))

    if original_hintergrund is None:
        print("Fehler: 'no_container.jpg' nicht gefunden!")
        return None

    # Ziel-Dimensionen berechnen und Hintergrund skalieren
    original_breite = int(original_hintergrund.shape[1] * scale_factor)
    original_hoehe = int(original_hintergrund.shape[0] * scale_factor)
    hintergrund = cv2.resize(original_hintergrund, (original_breite, original_hoehe), interpolation=cv2.INTER_AREA)

    # Fokusbereiche (ROIs) der einzelnen Regalfächer aus den Enums laden
    fokusbereiche = ENUM.FOKUSBEREICHE

    # SPS-Bestandsdaten extrahieren
    stock_items = sps_daten["payload"]["stockItems"]

    # Debug-Ausgabe aller empfangenen SPS-Einträge
    for item in stock_items:
        print(item)

    # Array zum gegenprüfen in der finalen auswertung. Enthält die Zustände, die das Kamerasystem tatsächlich hätte sehen sollen
    tatsaechlicher_zustand = []

    # --- 2. VERARBEITUNG DER EINZELNEN REGALFÄCHER ---
    for idx, fach in enumerate(stock_items):
        # Prüfen, ob das Fach mit einem Werkstück belegt ist
        tatsaechlicher_zustand.append({"Belegung": fach['workpiece']['type'], "Anomalien": []})
        if fach["workpiece"]["type"] != "":
            reihe = fach['location'][0]  # Zeilenbezeichnung (z. B. 'A', 'B', 'C')
            farbe = fach['workpiece']['type']  # Werkstückfarbe

            # Basisbild-Referenz aus der Mapping-Tabelle ermitteln
            quellbild = ENUM.BASISBILDER[reihe][farbe]

            # Quellbild laden, falls ein Dateipfad-String übergeben wurde
            if isinstance(quellbild, str):
                quellbild = cv2.imread(os.path.join(datenbank_pfad, quellbild))
            else:
                quellbild = quellbild

            # Validierung, ob das Bild geladen werden konnte
            if quellbild is None:
                print(f"Fehler: Bild für Reihe {reihe}, Farbe {farbe} konnte nicht geladen werden!")
                continue

            # Quellbild auf die globale Zielgröße skalieren
            quellbild = cv2.resize(quellbild, (original_breite, original_hoehe), interpolation=cv2.INTER_AREA)
            print(f"Fach {idx} ({fach['location']}): {farbe}")

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
    cv2.imwrite("data/img/current.jpg", hintergrund)
    print("-> Neues Testbild erfolgreich skaliert und als 'data/img/current.jpg' gespeichert.")

    return tatsaechlicher_zustand