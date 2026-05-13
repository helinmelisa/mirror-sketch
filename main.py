import cv2
import numpy as np
from filters import dodge_blend, xdog, canny_sketch, hatching, etf_sketch, apply_paper_texture

MODES = ["dodge", "xdog", "canny", "hatching", "etf"]
MODE_LABELS = {
    "dodge":    "Pencil Sketch (Dodge)",
    "xdog":     "XDoG (Artistic)",
    "canny":    "Canny Stylized",
    "hatching": "Hatching",
    "etf":      "ETF Coherent Strokes",
}

KEYS = {
    ord("1"): "dodge",
    ord("2"): "xdog",
    ord("3"): "canny",
    ord("4"): "hatching",
    ord("5"): "etf",
    ord("t"): "toggle_texture",
    ord("m"): "mirror",
    ord("q"): "quit",
    27:        "quit",  # ESC
}


def draw_hud(frame: np.ndarray, mode: str, texture: bool, mirrored: bool) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    label = f"[{MODES.index(mode)+1}] {MODE_LABELS[mode]}  |  "
    label += f"[T] Texture: {'on' if texture else 'off'}  |  "
    label += f"[M] Mirror: {'on' if mirrored else 'off'}  |  [Q] Quit"
    cv2.putText(frame, label, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def process(gray: np.ndarray, mode: str, texture: bool) -> np.ndarray:
    if mode == "dodge":
        out = dodge_blend(gray)
    elif mode == "xdog":
        out = xdog(gray)
    elif mode == "canny":
        out = canny_sketch(gray)
    elif mode == "hatching":
        out = hatching(gray)
    elif mode == "etf":
        out = etf_sketch(gray)
    else:
        out = gray

    if texture:
        out = apply_paper_texture(out)

    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mode = "xdog"
    texture = True
    mirrored = True

    cv2.namedWindow("Mirror Sketch", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mirrored:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = process(gray, mode, texture)
        draw_hud(output, mode, texture, mirrored)

        cv2.imshow("Mirror Sketch", output)

        key = cv2.waitKey(1) & 0xFF
        action = KEYS.get(key)

        if action == "quit":
            break
        elif action == "toggle_texture":
            texture = not texture
        elif action == "mirror":
            mirrored = not mirrored
        elif action in MODES:
            mode = action

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
