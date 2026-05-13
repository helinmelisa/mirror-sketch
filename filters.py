import cv2
import numpy as np
from etf import compute_tangent_field, lic, fdog


def dodge_blend(gray: np.ndarray, blur_radius: int = 21) -> np.ndarray:
    inv = 255 - gray
    blurred = cv2.GaussianBlur(inv, (blur_radius | 1, blur_radius | 1), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256.0)
    return np.clip(sketch, 0, 255).astype(np.uint8)


def xdog(gray: np.ndarray, sigma: float = 0.5, k: float = 4.5,
         p: float = 19.0, epsilon: float = -0.1, phi: float = 10.0) -> np.ndarray:
    g1 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma * k)
    dog = g1 - (1.0 / k) * g2
    # soft thresholding
    result = np.where(dog >= epsilon, 1.0, 1.0 + np.tanh(phi * (dog - epsilon)))
    result = (result * 255).clip(0, 255).astype(np.uint8)
    return result


def canny_sketch(gray: np.ndarray, low: int = 30, high: int = 90,
                 blur: int = 5) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (blur | 1, blur | 1), 0)
    edges = cv2.Canny(blurred, low, high)
    return 255 - edges


def hatching(gray: np.ndarray, angle_deg: float = 45.0,
             spacing: int = 6, thickness: int = 1) -> np.ndarray:
    h, w = gray.shape
    canvas = np.ones((h, w), dtype=np.uint8) * 255
    angle = np.deg2rad(angle_deg)
    dx, dy = np.cos(angle), np.sin(angle)

    for i in range(-max(h, w), max(h, w), spacing):
        x0 = int(i * (-dy))
        y0 = int(i * dx)
        x1 = int(x0 + w * dx)
        y1 = int(y0 + w * dy)
        cv2.line(canvas, (x0, y0), (x1, y1), 0, thickness)

    # only show hatch where image is dark
    mask = (gray < 128).astype(np.uint8) * 255
    result = np.where(mask > 0, canvas, 255).astype(np.uint8)
    return result


def etf_sketch(gray: np.ndarray, lic_steps: int = 12, fdog_steps: int = 8,
               scale: float = 0.5) -> np.ndarray:
    """Coherent stroke sketch via ETF tangent field + FDoG edges."""
    h, w = gray.shape
    sh, sw = int(h * scale), int(w * scale)

    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    tx, ty, mag = compute_tangent_field(small)

    # LIC on noise — fibrous texture aligned with edges
    noise = np.random.randint(0, 256, (sh, sw), dtype=np.uint8)
    strokes = lic(noise, tx, ty, steps=lic_steps)

    # FDoG edges along the flow
    edges = fdog(small, tx, ty, steps=fdog_steps)

    # Multiply: strokes darken where edges are strong
    mag_up = cv2.resize(mag.astype(np.float32), (sw, sh))
    combined = (strokes.astype(np.float32) * (edges.astype(np.float32) / 255.0))
    combined = np.clip(combined, 0, 255).astype(np.uint8)

    return cv2.resize(combined, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_paper_texture(sketch: np.ndarray, strength: float = 0.15) -> np.ndarray:
    noise = np.random.normal(0, 255 * strength, sketch.shape).astype(np.float32)
    textured = np.clip(sketch.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return textured
