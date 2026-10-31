"""Shared helpers: image loading, cell masks and intensity normalisation."""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "results"


def load_gray(path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.float32)


def save_gray(arr: np.ndarray, path) -> None:
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def cell_mask(img: np.ndarray, sigma: float = 3) -> np.ndarray:
    """Largest bright connected region (the pouch-cell body)."""
    s = ndi.gaussian_filter(img.astype(float), sigma)
    m = s > threshold_otsu(s)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    m = lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)
    return ndi.binary_fill_holes(ndi.binary_closing(m, iterations=3))


def fov_mask(img: np.ndarray) -> np.ndarray:
    """Pixels inside the reconstructed field of view (excludes the black
    corners outside the circular FOV of the bf_* scans)."""
    s = ndi.gaussian_filter(img.astype(float), 2)
    m = ndi.binary_fill_holes(s > 8)
    return ndi.binary_erosion(m, iterations=3)


def norm_params(img: np.ndarray, valid: np.ndarray | None = None):
    """Map background -> 0 and bright cell material -> 1.

    Background = median of a ring just outside the cell; bright level =
    99th percentile inside the cell.  Uses only image content, so the same
    rule is applied to synthetic training inputs and to the real bf_ scans.
    """
    m = cell_mask(img)
    ring = ndi.binary_dilation(m, iterations=25) & ~ndi.binary_dilation(m, iterations=8)
    if valid is not None:
        ring &= valid
    lo = float(np.median(img[ring])) if ring.any() else float(np.percentile(img, 5))
    hi = float(np.percentile(img[m], 99)) if m.any() else float(img.max())
    return lo, max(hi - lo, 1e-3)


def normalise(img, lo, scale):
    return (img - lo) / scale


def denormalise(x, lo, scale):
    return x * scale + lo
