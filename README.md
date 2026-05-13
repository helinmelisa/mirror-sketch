# mirror-sketch

Real-time pencil sketch mirror using your webcam. See yourself as a live drawing.

## Sketch Modes

| Key | Mode |
|-----|------|
| `1` | Pencil Sketch — classic dodge blend |
| `2` | XDoG — artistic ink sketch |
| `3` | Canny — hard edge / graphic style |
| `4` | Hatching — directional line overlay |
| `5` | ETF — coherent strokes via Edge Tangent Flow + FDoG |

## Effects

| Key | Effect |
|-----|--------|
| `T` | Paper texture noise |
| `F` | Stroke flicker — hand-drawn jitter via random affine warp |
| `R` | Motion trails — optical flow leaves fading sketch echoes |
| `C` | Color sketch — preserves original hue, sketch drives luminance |
| `M` | Mirror flip |
| `Q` / ESC | Quit |

Effects stack — try ETF + Color + Trails + Flicker together.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Requirements

- Python 3.8+
- Webcam
