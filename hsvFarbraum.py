import cv2
import numpy as np

# --- Schritt 0: Bild laden und skalieren ---
image = cv2.imread("data/img/current.jpg")
if image is None:
    print("Fehler: Bild konnte nicht geladen werden.")
    exit()

scale_factor = 0.3
width = int(image.shape[1] * scale_factor)
height = int(image.shape[0] * scale_factor)
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# --- Schritt 1: ROIs manuell definieren ---
rois = [
    (180, 285, 210, 315),    # Container 1
    (180, 285, 535, 640),   # Container 2
    (180, 285, 850, 955),   # Container 3
    (370, 475, 210, 315),   # Container 4
    (370, 475, 535, 640),   # Container 5
    (370, 475, 840, 945),   # Container 6
    (560, 665, 210, 315),   # Container 7
    (560, 665, 535, 640),   # Container 8
    (560, 665, 830, 935)    # Container 9
]

# --- Schritt 2: HSV-Werte der Steine extrahieren ---
for idx, (y_start, y_end, x_start, x_end) in enumerate(rois):
    print(f"\n--- Container {idx + 1} ---")

    # ROI extrahieren
    roi = image[y_start:y_end, x_start:x_end]

    # Hintergrund ausschließen: Nur inneren Bereich analysieren
    top_margin = 0.1  # 10% Rand oben ausschließen
    bottom_margin = 0.5  # 10% Rand unten ausschließen
    inner_y_start = int((y_end - y_start) * top_margin)
    inner_y_end = int((y_end - y_start) * (1 - bottom_margin))
    inner_roi = roi[inner_y_start:inner_y_end, :]  # Von inner_y_start bis

    inner_roi_hsv = cv2.cvtColor(inner_roi, cv2.COLOR_BGR2HSV)

    # HSV-Kanäle trennen
    h, s, v = cv2.split(inner_roi_hsv)

    # Min/Max-Werte für jeden Kanal
    print(f"Hue: Min={h.min()}, Max={h.max()}, Mittelwert={h.mean():.1f}")
    print(f"Saturation: Min={s.min()}, Max={s.max()}, Mittelwert={s.mean():.1f}")
    print(f"Value: Min={v.min()}, Max={v.max()}, Mittelwert={v.mean():.1f}")

    # Histogramm für jeden Kanal erstellen
    hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])

    # Histogramm visualisieren (einfach als Bild)
    hist_img = np.zeros((300, 256, 3), dtype=np.uint8)
    for i in range(256):
        # Hue (Blau)
        cv2.line(hist_img, (i, 100), (i, 100 - int(hist_h[i][0] * 0.01)), (255, 0, 0), 1)
        # Saturation (Grün)
        cv2.line(hist_img, (i, 200), (i, 200 - int(hist_s[i][0] * 0.01)), (0, 255, 0), 1)
        # Value (Rot)
        cv2.line(hist_img, (i, 300), (i, 300 - int(hist_v[i][0] * 0.01)), (0, 0, 255), 1)

    # cv2.imshow(f"Histogramm HSV für Container {idx + 1}", hist_img)
    # cv2.waitKey(0)

    # ROI und inneren Bereich anzeigen
    # cv2.imshow(f"ROI Container {idx + 1}", roi)
    # cv2.imshow(f"Innerer Bereich Container {idx + 1}", inner_roi)
    # cv2.waitKey(0)

cv2.destroyAllWindows()