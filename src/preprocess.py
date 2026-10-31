"""Prepare the raw XCT slices for analysis.

- bf_*  (blurred inputs): strip the coloured viewer border lines, convert to
  8-bit grayscale, and save a binary mask of the cell body.
- at_*  (sharp references): convert to grayscale and mask out the scale bar.

Usage:  python src/preprocess.py
Outputs go to data/processed/.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"

# Crop box (left, top, right, bottom) that removes the red/blue/green
# border lines drawn by the viewer in the bf_* exports.
BF_CROP = (17, 4, 451, 313)
# Region of the at_* images holding the "5 mm" scale bar (x0, y0) -> to edge.
AT_SCALEBAR_FROM = (370, 112)


def to_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA").convert("L"))


def cell_mask(img: np.ndarray) -> np.ndarray:
    """Largest bright connected region = the pouch-cell body."""
    smooth = ndi.gaussian_filter(img.astype(float), 3)
    m = smooth > threshold_otsu(smooth)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (np.argmax(sizes) + 1)
    return ndi.binary_fill_holes(ndi.binary_closing(m, iterations=3))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in sorted(RAW.glob("bf_*.png")):
        g = to_gray(p)
        l, t, r, b = BF_CROP
        g = g[t:b, l:r]
        Image.fromarray(g).save(OUT / p.name)
        Image.fromarray((cell_mask(g) * 255).astype(np.uint8)).save(
            OUT / p.name.replace(".png", "_mask.png"))
        print(f"{p.name}: {g.shape}")
    for p in sorted(RAW.glob("at_*.png")):
        g = to_gray(p)
        x0, y0 = AT_SCALEBAR_FROM
        valid = np.ones_like(g, dtype=bool)
        valid[y0:, x0:] = False
        g = g.copy()
        g[~valid] = int(np.median(g[g < 40])) if (g < 40).any() else 0
        Image.fromarray(g).save(OUT / p.name)
        print(f"{p.name}: {g.shape}")


if __name__ == "__main__":
    main()
