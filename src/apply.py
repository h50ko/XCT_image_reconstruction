"""Apply the trained U-Net to the real bf_* scans.

For each bf_ image:
  1. normalise (background -> 0, bright cell -> 1) with the same rule used
     for the synthetic training inputs,
  2. fill the black area outside the circular field of view with background
     so the network only sees the kind of content it was trained on,
  3. restore, then de-normalise back to 8-bit and restore the outside-FOV
     pixels.
Also writes a Richardson-Lucy deconvolution baseline and comparison figures.

Usage:  python src/apply.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.restoration import richardson_lucy

from common import PROCESSED, RESULTS, denormalise, fov_mask, load_gray, norm_params, normalise, save_gray
from model import UNet

NAMES = ["bf_T0", "bf_T0_2", "bf_T0_3"]


def rl_psf(sy=3.5, sx=1.2, core=0.6, w=0.6, size=21):
    """Mean of the degradation model's PSF (core + vertical halo)."""
    ax = np.arange(size) - size // 2
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    g = lambda s_y, s_x: np.exp(-0.5 * ((yy / s_y) ** 2 + (xx / s_x) ** 2))
    k = (1 - w) * g(core, core) / g(core, core).sum() + w * g(sy, sx) / g(sy, sx).sum()
    return k / k.sum()


def restore_unet(net, x):
    with torch.no_grad():
        return net(torch.from_numpy(x.astype(np.float32))[None, None]).numpy()[0, 0]


def restore_rl(x, iters=15):
    return richardson_lucy(np.clip(x, 0, None), rl_psf(), num_iter=iters, clip=False)


def main():
    out_dir = RESULTS / "unet"
    net = UNet()
    net.load_state_dict(torch.load(out_dir / "checkpoints/best.pt", map_location="cpu"))
    net.eval()

    for name in NAMES:
        raw = load_gray(PROCESSED / f"{name}.png")
        fov = fov_mask(raw)
        lo, sc = norm_params(raw, fov)
        x = normalise(raw, lo, sc)
        x[~fov] = 0.0
        y_unet = restore_unet(net, x)
        y_rl = restore_rl(x)
        res = {}
        for tag, y in [("unet", y_unet), ("rl", y_rl)]:
            img = denormalise(np.clip(y, -0.1, 1.3), lo, sc)
            img[~fov] = raw[~fov]
            save_gray(img, out_dir / f"{name}_{tag}.png")
            res[tag] = img
        compare_figure(name, raw, res, out_dir / f"{name}_compare.png")
        print(f"{name}: wrote {name}_unet.png, {name}_rl.png, {name}_compare.png")


def compare_figure(name, raw, res, path):
    """Full view + 3 zooms (left end, middle, right end) for each method."""
    rows = [("input (bf)", raw), ("Richardson-Lucy", res["rl"]), ("U-Net (ours)", res["unet"])]
    zooms = [(100, 205, 0, 120), (100, 205, 160, 280), (100, 205, 314, 434)]
    fig, ax = plt.subplots(3, 4, figsize=(18, 9), gridspec_kw=dict(width_ratios=[2.2, 1, 1, 1]))
    for r, (label, img) in enumerate(rows):
        ax[r, 0].imshow(img[80:230], cmap="gray", vmin=0, vmax=255)
        ax[r, 0].set_ylabel(label, fontsize=12)
        for c, (y0, y1, x0, x1) in enumerate(zooms, 1):
            ax[r, c].imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    for c, t in enumerate(["full (cropped)", "left end", "middle", "right end"]):
        ax[0, c].set_title(t)
    fig.suptitle(name)
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
