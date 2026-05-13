import cv2
import numpy as np
from filters import (dodge_blend, xdog, canny_sketch, hatching,
                     etf_sketch, apply_paper_texture, apply_flicker, color_sketch)

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
    ord("f"): "toggle_flicker",
    ord("r"): "toggle_trails",
    ord("c"): "toggle_color",
    ord("m"): "toggle_mirror",
    ord("q"): "quit",
    27:        "quit",
}


class TrailBuffer:
    def __init__(self, decay: float = 0.82):
        self.decay = decay
        self.buf: np.ndarray | None = None
        self.prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self.buf = None
        self.prev_gray = None

    def apply(self, sketch_bgr: np.ndarray, gray: np.ndarray) -> np.ndarray:
        h, w = sketch_bgr.shape[:2]

        if self.buf is None or self.buf.shape != sketch_bgr.shape:
            self.buf = np.full_like(sketch_bgr, 255, dtype=np.float32)
            self.prev_gray = gray.copy()
            return sketch_bgr

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion = (mag > 1.5).astype(np.float32)[:, :, None]

        # decay trail toward white
        self.buf = self.buf * self.decay + 255.0 * (1.0 - self.decay)

        # stamp current sketch into trail where motion detected (darkest wins)
        self.buf = np.where(
            motion > 0,
            np.minimum(self.buf, sketch_bgr.astype(np.float32)),
            self.buf
        )

        self.prev_gray = gray.copy()

        # composite: darkest of current + trail
        result = np.minimum(sketch_bgr.astype(np.float32), self.buf)
        return result.clip(0, 255).astype(np.uint8)


def get_sketch(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "dodge":
        return dodge_blend(gray)
    elif mode == "xdog":
        return xdog(gray)
    elif mode == "canny":
        return canny_sketch(gray)
    elif mode == "hatching":
        return hatching(gray)
    elif mode == "etf":
        return etf_sketch(gray)
    return gray


def draw_hud(frame: np.ndarray, mode: str, flags: dict) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 70), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    line1 = "  ".join(f"[{i+1}] {MODE_LABELS[m]}" if m == mode
                      else f"[{i+1}] {m}" for i, m in enumerate(MODES))
    line2 = (f"[T] Texture:{'on' if flags['texture'] else 'off'}  "
             f"[F] Flicker:{'on' if flags['flicker'] else 'off'}  "
             f"[R] Trails:{'on' if flags['trails'] else 'off'}  "
             f"[C] Color:{'on' if flags['color'] else 'off'}  "
             f"[M] Mirror:{'on' if flags['mirror'] else 'off'}  "
             f"[Q] Quit")

    cv2.putText(frame, line1, (8, h - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (8, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mode = "xdog"
    flags = {"texture": True, "flicker": False, "trails": False,
             "color": False, "mirror": True}

    trails = TrailBuffer(decay=0.82)
    cv2.namedWindow("Mirror Sketch", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if flags["mirror"]:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sketch_gray = get_sketch(gray, mode)

        if flags["texture"]:
            sketch_gray = apply_paper_texture(sketch_gray)
        if flags["flicker"]:
            sketch_gray = apply_flicker(sketch_gray)

        if flags["color"]:
            output = color_sketch(frame, sketch_gray)
        else:
            output = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)

        if flags["trails"]:
            output = trails.apply(output, gray)
        else:
            trails.reset()

        draw_hud(output, mode, flags)
        cv2.imshow("Mirror Sketch", output)

        key = cv2.waitKey(1) & 0xFF
        action = KEYS.get(key)

        if action == "quit":
            break
        elif action == "toggle_texture":
            flags["texture"] = not flags["texture"]
        elif action == "toggle_flicker":
            flags["flicker"] = not flags["flicker"]
        elif action == "toggle_trails":
            flags["trails"] = not flags["trails"]
        elif action == "toggle_color":
            flags["color"] = not flags["color"]
        elif action == "toggle_mirror":
            flags["mirror"] = not flags["mirror"]
        elif action in MODES:
            mode = action

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
