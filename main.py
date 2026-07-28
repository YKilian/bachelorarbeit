import datetime
import random
import createImage
import pipeline


def uebersetze_sps_daten(sps_daten: dict) -> list:
    """Übersetzt die SPS-Payload in das interne Soll-Zustandsformat."""
    zustand = []
    stock_items = sps_daten["payload"]["stockItems"]
    for stock_item in stock_items:
        zustand.append({
            "Belegung": stock_item["workpiece"]["type"],
            "Anomalien": []
        })
    return zustand


def erstelle_zufaellige_sps_daten(possible_states: list) -> dict:
    """Generiert ein dynamisches SPS-Daten-Dictionary mit zufälliger Belegung."""
    now = datetime.datetime.now()
    locations = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]

    return {
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "topic": "ccu/state/stock",
        "payload": {
            "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stockItems": [
                {
                    "workpiece": {"id": "", "type": random.choice(possible_states), "state": "RAW"},
                    "location": loc,
                    "hbw": "SVR4H73901"
                }
                for loc in locations
            ]
        }
    }


def drucke_metriken_bericht(metriken: dict, moegliche_anomalien: list):
    """Erstellt einen übersichtlichen wissenschaftlichen Evaluierungsbericht."""
    print("\n" + "=" * 70)
    print("      EVALUIERUNGSBERICHT DER ERKENNUNGS-PIPELINE")
    print("=" * 70)
    print(f"{'Anomalie':<20} | {'RP':<8} | {'FP':<8} | {'RN':<8} | {'FN':<8}")
    print("-" * 70)

    for anomalie in moegliche_anomalien:
        rp = metriken["RP"][anomalie]
        fp = metriken["FP"][anomalie]
        rn = metriken["RN"][anomalie]
        fn = metriken["FN"][anomalie]
        print(f"{anomalie:<20} | {rp:<8} | {fp:<8} | {rn:<8} | {fn:<8}")

    print("-" * 70)
    print(f"{'Anomalie':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 70)

    for anomalie in moegliche_anomalien:
        rp = metriken["RP"][anomalie]
        fp = metriken["FP"][anomalie]
        fn = metriken["FN"][anomalie]

        precision = rp / (rp + fp) if (rp + fp) > 0 else 0.0
        recall = rp / (rp + fn) if (rp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"{anomalie:<20} | {precision * 100:>8.1f}% | {recall * 100:>8.1f}% | {f1 * 100:>8.1f}%")
    print("=" * 70 + "\n")


def main():
    POSSIBLE_STATES = [""]
    MOEGLICHE_ANOMALIEN = ["Farbe", "Verkantung", "Container_Rotiert", "Container_Fehlt"]

    # 1. Testdaten generieren
    sps_daten = erstelle_zufaellige_sps_daten(POSSIBLE_STATES)
    soll_zustand = uebersetze_sps_daten(sps_daten)

    # 2. Testbild rendern & Pipeline ausführen
    # createImage.generiere_sps_state gibt das OpenCV-Bild zurück und speichert current.jpg
    testbild = createImage.generiere_sps_state(sps_daten, generate_error=False)

    # Physischer Ist-Zustand (Entspricht bei generierten Bildern zunächst dem Soll-Zustand)
    physischer_zustand = soll_zustand

    # Pipeline-Analyse aufrufen
    gesehener_zustand = pipeline.main(soll_zustand)

    # 3. Konsolenausgabe: Einzelvalidierung pro Fach
    print("\n=== VALIDIERUNG GEGEN TEST-LAYOUT ===")
    header_fmt = "{:<10} | {:<20} | {:<20} | {:<20} | {:<10}"
    print(header_fmt.format("Container", "Physisch (Ist)", "Erkannt (Gesehen)", "System (Soll)", "Status"))
    print("-" * 90)

    # Metriken-Container initialisieren
    metriken = {
        kategorie: {anomalie: 0 for anomalie in MOEGLICHE_ANOMALIEN}
        for kategorie in ["RP", "FP", "RN", "FN"]
    }

    # 4. Auswertung pro Fach & Metriken-Berechnung
    for idx, fach in enumerate(soll_zustand):
        ist = physischer_zustand[idx]
        gesehen = gesehener_zustand[idx]
        soll = soll_zustand[idx]

        match_status = "✅ OK" if ist == gesehen else "❌ FALSCH"

        ist_str = f"{ist['Belegung']} {ist['Anomalien']}"
        gesehen_str = f"{gesehen['Belegung']} {gesehen['Anomalien']}"
        soll_str = f"{soll['Belegung']} {soll['Anomalien']}"

        print(header_fmt.format(f"Fach {idx}", ist_str, gesehen_str, soll_str, match_status))

        # Konfusionsmatrix für alle Anomaliearten befüllen
        for anomalie in MOEGLICHE_ANOMALIEN:
            in_ist = anomalie in ist['Anomalien']
            in_gesehen = anomalie in gesehen['Anomalien']

            if in_ist and in_gesehen:
                metriken["RP"][anomalie] += 1  # Real-Positiv (True Positive)
            elif in_ist and not in_gesehen:
                metriken["FN"][anomalie] += 1  # Falsch-Negativ (Missed Detection)
            elif not in_ist and in_gesehen:
                metriken["FP"][anomalie] += 1  # Falsch-Positiv (False Alarm)
            else:
                metriken["RN"][anomalie] += 1  # Real-Negativ (True Negative)

    # 5. Abschlussbericht drucken
    drucke_metriken_bericht(metriken, MOEGLICHE_ANOMALIEN)

    return metriken

def sum_nested_dicts(*dicts):
    result = {}
    for d in dicts:
        for outer_key in d:
            if outer_key not in result:
                result[outer_key] = {}
            for inner_key in d[outer_key]:
                if inner_key not in result[outer_key]:
                    result[outer_key][inner_key] = 0
                result[outer_key][inner_key] += d[outer_key][inner_key]
    return result

if __name__ == "__main__":
    overall_metriken = {}
    for i in range(1):
        metriken = main()
        overall_metriken = sum_nested_dicts(overall_metriken, metriken)

    print(overall_metriken)