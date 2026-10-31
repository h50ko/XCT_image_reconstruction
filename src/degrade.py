"""Synthetic degradation model: sharp at_* image -> bf_*-like image.

Designed from measurements of the bf_* scans (see README, "Degradation model"):

* Cell boundaries in the flat section ramp over ~10-15 px vertically
  (vs 1-2 px in at_*), yet a weak ~2.6 px layer ripple survives.  A single
  Gaussian would erase that ripple, so the PSF is a *narrow core plus a wide,
  mostly vertical halo*:  PSF = w * G(core) + (1 - w) * G(halo_x, halo_y).
* The blur is stronger in the middle of the cell than at the rounded ends,
  so the halo weight varies smoothly along x.
* Lower contrast and a raised background (haze).
* Horizontal streaks (noise correlated along x) and white noise.

All parameters are drawn at random per call so the network learns to handle
a range of degradations (blind restoration), including almost none, which
teaches it to leave already-sharp regions alone.
"""
import numpy as np
from scipy import ndimage as ndi


def sample_params(rng: np.random.Generator) -> dict:
    return dict(
        core_sigma=rng.uniform(0.3, 0.9),
        halo_sx=rng.uniform(0.5, 2.0),
        halo_sy=rng.uniform(2.0, 5.5),
        halo_w_min=rng.uniform(0.0, 0.4),   # halo weight at the cell ends
        halo_w_max=rng.uniform(0.5, 0.95),  # halo weight in the middle
        contrast=rng.uniform(0.45, 0.9),
        haze=rng.uniform(0.03, 0.2),
        streak_amp=rng.uniform(0.0, 0.06),
        streak_len=rng.uniform(4, 20),
        noise=rng.uniform(0.005, 0.04),
        gamma=rng.uniform(0.8, 1.25),
    )


def degrade(x: np.ndarray, rng: np.random.Generator, p: dict | None = None) -> np.ndarray:
    """x: clean image normalised to ~[0, 1] (background 0, cell ~1)."""
    p = p or sample_params(rng)
    h, w = x.shape
    core = ndi.gaussian_filter(x, p["core_sigma"])
    halo = ndi.gaussian_filter(x, (p["halo_sy"], p["halo_sx"]))
    # Halo weight peaks somewhere in the middle and falls off towards the ends.
    xs = np.linspace(-1, 1, w)
    centre = rng.uniform(-0.3, 0.3)
    prof = np.clip(1 - (np.abs(xs - centre) / rng.uniform(0.7, 1.2)) ** 2, 0, 1)
    wgt = p["halo_w_min"] + (p["halo_w_max"] - p["halo_w_min"]) * prof
    y = (1 - wgt)[None, :] * core + wgt[None, :] * halo
    # Contrast loss + haze.
    y = p["haze"] + p["contrast"] * y
    # Horizontal streaks: white noise smeared along x.
    streaks = ndi.gaussian_filter(rng.standard_normal((h, w)), (0.6, p["streak_len"]))
    streaks /= streaks.std() + 1e-8
    y = y + p["streak_amp"] * streaks * np.clip(y, 0, None)
    y = y + p["noise"] * rng.standard_normal((h, w))
    y = np.sign(y) * np.abs(y) ** p["gamma"]
    return y.astype(np.float32)
