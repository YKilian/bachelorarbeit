import pipeline
import createImage
import numpy as np

if __name__ == "__main__":
    test_layout = [
        ["wrb_ok", "wrb_ok", "wrb_ok"],
        ["wrb_ok", "wrb_ok", "wrb_ok"],
        ["wrb_ok", "wrb_ok", "wrb_ok"]
    ]

    # Das 3x3 Layout flach klopfen, damit wir per Index (0-8) darauf zugreifen können
    soll_liste = np.array(test_layout).flatten()

    createImage.generiere_test_zustand(test_layout)
    cv_saw = pipeline.main()

    print("\n=== VALIDIERUNG GEGEN TEST-LAYOUT ===")
    print("{:<12} {:<15} {:<15} {:<10}".format("Container", "Erkannt (Ist)", "Layout (Soll)", "Status"))
    print("-" * 55)

    for idx, container in enumerate(cv_saw):
        # 1. Erkannte Werte übersetzen / mappen
        match container["Farbe"]:
            case "rot":
                if container["Container"] in [1, 2, 3]:
                    container["Farbe"] = "rbw"
                elif container["Container"] in [4, 5, 6]:
                    container["Farbe"] = "wrb"
                elif container["Container"] in [7, 8, 9]:
                    container["Farbe"] = "bwr"
            case "weiss":
                if container["Container"] in [1, 2, 3]:
                    container["Farbe"] = "wrb"
                elif container["Container"] in [4, 5, 6]:
                    container["Farbe"] = "bwr"
                elif container["Container"] in [7, 8, 9]:
                    container["Farbe"] = "rbw"
            case "blau":
                if container["Container"] in [1, 2, 3]:
                    container["Farbe"] = "bwr"
                elif container["Container"] in [4, 5, 6]:
                    container["Farbe"] = "rbw"
                elif container["Container"] in [7, 8, 9]:
                    container["Farbe"] = "wrb"
            case "leer":
                container["Farbe"] = "leer"
                container["Lage"] = ""

        # 2. Den erkannten String (Ist-Zustand) zusammenbauen
        if container["Farbe"] == "leer":
            ist_string = "leer"
        else:
            # Da deine Pipeline 'ok' oder 'verkantet' liefert, dein Layout aber '_ok' und '_error' nutzt:
            lage_gemappt = "ok" if container["Lage"] == "ok" else "error"
            ist_string = f"{container['Farbe']}_{lage_gemappt}"

        # 3. Den erwarteten String (Soll-Zustand) aus der Liste holen
        soll_string = soll_liste[idx]

        # 4. Abgleich durchführen
        if ist_string == soll_string:
            match_status = "✅ OK"
        else:
            match_status = "❌ FALSCH"

        # Ausgabe pro Container
        print("{:<12} {:<15} {:<15} {:<10}".format(
            f"Fach {container['Container']}",
            ist_string,
            soll_string,
            match_status
        ))