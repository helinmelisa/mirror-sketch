import cv2
import numpy as np


def compute_tangent_field(gray: np.ndarray, smooth_sigma: float = 2.0) -> tuple:
    """Structure tensor → dominant tangent direction per pixel."""
    f = gray.astype(np.float32)

    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=5)

    # Structure tensor components
    k = int(smooth_sigma * 4) | 1
    Jxx = cv2.GaussianBlur(gx * gx, (k, k), smooth_sigma)
    Jxy = cv2.GaussianBlur(gx * gy, (k, k), smooth_sigma)
    Jyy = cv2.GaussianBlur(gy * gy, (k, k), smooth_sigma)

    # Minor eigenvector = tangent along edge
    theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
    tx = np.cos(theta)
    ty = np.sin(theta)

    # Edge strength (coherence proxy)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = cv2.GaussianBlur(mag, (k, k), smooth_sigma)
    mag /= mag.max() + 1e-8

    return tx, ty, mag


def lic(texture: np.ndarray, tx: np.ndarray, ty: np.ndarray,
        steps: int = 15) -> np.ndarray:
    """Line Integral Convolution along tangent field."""
    h, w = texture.shape
    result = texture.astype(np.float32).copy()
    weight_sum = np.ones((h, w), np.float32)

    cur_xf = np.arange(w, dtype=np.float32)[None].repeat(h, 0)
    cur_yf = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
    cur_xb, cur_yb = cur_xf.copy(), cur_yf.copy()

    for i in range(1, steps + 1):
        w_i = 1.0 / (i + 1)

        # forward
        ix = np.clip(cur_xf, 0, w - 1).astype(np.int32)
        iy = np.clip(cur_yf, 0, h - 1).astype(np.int32)
        cur_xf = np.clip(cur_xf + tx[iy, ix], 0, w - 1)
        cur_yf = np.clip(cur_yf + ty[iy, ix], 0, h - 1)
        result += w_i * texture[cur_yf.astype(np.int32), cur_xf.astype(np.int32)]
        weight_sum += w_i

        # backward
        ix = np.clip(cur_xb, 0, w - 1).astype(np.int32)
        iy = np.clip(cur_yb, 0, h - 1).astype(np.int32)
        cur_xb = np.clip(cur_xb - tx[iy, ix], 0, w - 1)
        cur_yb = np.clip(cur_yb - ty[iy, ix], 0, h - 1)
        result += w_i * texture[cur_yb.astype(np.int32), cur_xb.astype(np.int32)]
        weight_sum += w_i

    return (result / weight_sum).clip(0, 255).astype(np.uint8)


def fdog(gray: np.ndarray, tx: np.ndarray, ty: np.ndarray,
         sigma_c: float = 1.0, sigma_s: float = 1.6,
         sigma_m: float = 3.0, tau: float = 0.98,
         steps: int = 10) -> np.ndarray:
    """Flow-based Difference of Gaussians — edges follow the tangent field."""
    h, w = gray.shape
    f = gray.astype(np.float32)

    # Gradient-aligned DoG (sampled perpendicular to tangent = along gradient)
    nx, ny = -ty, tx  # normal direction

    g1 = cv2.GaussianBlur(f, (0, 0), sigma_c)
    g2 = cv2.GaussianBlur(f, (0, 0), sigma_s)
    dog_base = g1 - tau * g2

    # Smooth dog_base along tangent (flow-aligned)
    result = dog_base.copy()
    weight_sum = np.ones((h, w), np.float32)

    cur_xf = np.arange(w, dtype=np.float32)[None].repeat(h, 0)
    cur_yf = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
    cur_xb, cur_yb = cur_xf.copy(), cur_yf.copy()

    for i in range(1, steps + 1):
        t = i / steps
        w_i = np.exp(-0.5 * (i / sigma_m) ** 2)

        ix = np.clip(cur_xf, 0, w - 1).astype(np.int32)
        iy = np.clip(cur_yf, 0, h - 1).astype(np.int32)
        cur_xf = np.clip(cur_xf + tx[iy, ix], 0, w - 1)
        cur_yf = np.clip(cur_yf + ty[iy, ix], 0, h - 1)
        result += w_i * dog_base[cur_yf.astype(np.int32), cur_xf.astype(np.int32)]
        weight_sum += w_i

        ix = np.clip(cur_xb, 0, w - 1).astype(np.int32)
        iy = np.clip(cur_yb, 0, h - 1).astype(np.int32)
        cur_xb = np.clip(cur_xb - tx[iy, ix], 0, w - 1)
        cur_yb = np.clip(cur_yb - ty[iy, ix], 0, h - 1)
        result += w_i * dog_base[cur_yb.astype(np.int32), cur_xb.astype(np.int32)]
        weight_sum += w_i

    result /= weight_sum

    # Soft thresholding
    phi = 10.0
    out = np.where(result >= 0, 1.0, 1.0 + np.tanh(phi * result))
    return (out * 255).clip(0, 255).astype(np.uint8)
