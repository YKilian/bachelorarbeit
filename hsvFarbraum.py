import cv2
import numpy as np

# --- 1. Bild laden und skalieren ---
IMAGE_PATH = "data/img/current.jpg"
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Fehler: Bild '{IMAGE_PATH}' konnte nicht geladen werden.")
    exit()

SCALE_FACTOR = 0.3
width = int(image.shape[1] * SCALE_FACTOR)
height = int(image.shape[0] * SCALE_FACTOR)
resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# In HSV konvertieren
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

# Callback-Funktion für die Regler (wird bei jeder Änderung aufgerufen)
def nothing(x):
    pass

# --- 2. Fenster und Trackbars erstellen ---
cv2.namedWindow("HSV Einstellungen", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HSV Einstellungen", 400, 300)

# Trackbars für LOWER Vektor
cv2.createTrackbar("Lower H", "HSV Einstellungen", 0, 180, nothing)
cv2.createTrackbar("Lower S", "HSV Einstellungen", 0, 255, nothing)
cv2.createTrackbar("Lower V", "HSV Einstellungen", 65, 255, nothing)

# Trackbars für UPPER Vektor
cv2.createTrackbar("Upper H", "HSV Einstellungen", 180, 180, nothing)
cv2.createTrackbar("Upper S", "HSV Einstellungen", 95, 255, nothing)
cv2.createTrackbar("Upper V", "HSV Einstellungen", 146, 255, nothing)

print("=" * 60)
print(" HSV SCHIEBEREGLER - TOOL")
print("=" * 60)
print("Stelle die Regler im Fenster 'HSV Einstellungen' ein.")
print("Drücke 'q' oder 'ESC' im Bildfenster, um zu beenden.")
print("=" * 60)

while True:
    # 1. Aktuelle Werte der Schieberegler auslesen
    l_h = cv2.getTrackbarPos("Lower H", "HSV Einstellungen")
    l_s = cv2.getTrackbarPos("Lower S", "HSV Einstellungen")
    l_v = cv2.getTrackbarPos("Lower V", "HSV Einstellungen")

    u_h = cv2.getTrackbarPos("Upper H", "HSV Einstellungen")
    u_s = cv2.getTrackbarPos("Upper S", "HSV Einstellungen")
    u_v = cv2.getTrackbarPos("Upper V", "HSV Einstellungen")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])

    # 2. Maske berechnen
    mask = cv2.inRange(hsv, lower, upper)

    # 3. Live-Text auf dem Originalbild einblenden
    text_canvas = resized.copy()
    cv2.putText(text_canvas, f"LOWER: {lower.tolist()}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(text_canvas, f"UPPER: {upper.tolist()}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 4. Fenster anzeigen
    cv2.imshow("Original mit Werten", text_canvas)
    cv2.imshow("HSV Maske Live", mask)

    # Beenden mit Tastendruck 'q' oder ESC (27)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == 27:
        break

# --- Finaler Code-Export ---
print("\n" + "=" * 60)
print(" DEINE FINALEN WERTE FÜR DEN CODE:")
print("=" * 60)
print(f"lower_black = np.array([{l_h}, {l_s}, {l_v}])")
print(f"upper_black = np.array([{u_h}, {u_s}, {u_v}])")
print("=" * 60)

cv2.destroyAllWindows()