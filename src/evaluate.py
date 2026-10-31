"""Score the trained U-Net on the held-out report slices.

Each held-out sharp slice is degraded with fixed seeds (same as training
validation), then restored by (a) Richardson-Lucy deconvolution and (b) the
U-Net.  PSNR / SSIM are computed against the clean slice on non-overlay
pixels.  Writes results/unet/val_metrics.json and val_examples.png.

Usage:  python src/evaluate.py [--holdout Report-2026-03-27_r2]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity

from apply import restore_rl, restore_unet
from common import RESULTS
from model import UNet
from train import load_targets, make_val_set, psnr


def ssim(p, t, m):
    _, full = structural_similarity(np.clip(p, 0, 1.2), t, data_range=1.2, full=True)
    return float(full[m].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="Report-2026-03-27_r2")
    args = ap.parse_args()
    out_dir = RESULTS / "unet"
    net = UNet()
    net.load_state_dict(torch.load(out_dir / "checkpoints/best.pt", map_location="cpu"))
    net.eval()

    _, val = load_targets(args.holdout)
    vset = make_val_set(val)
    scores = {k: {"psnr": [], "ssim": []} for k in ["input", "richardson_lucy", "unet"]}
    examples = []
    for inp, tgt, v in vset:
        outs = {"input": inp, "richardson_lucy": restore_rl(inp), "unet": restore_unet(net, inp)}
        for k, o in outs.items():
            scores[k]["psnr"].append(psnr(o, tgt, v))
            scores[k]["ssim"].append(ssim(o, tgt, v))
        examples.append((outs, tgt))
    summary = {k: {m: float(np.mean(x)) for m, x in d.items()} for k, d in scores.items()}
    summary["n_val_images"] = len(vset)
    summary["holdout"] = args.holdout
    (out_dir / "val_metrics.json").write_text(json.dumps(summary, indent=1))
    for k in ["input", "richardson_lucy", "unet"]:
        print(f"{k:16s} PSNR {summary[k]['psnr']:.2f} dB   SSIM {summary[k]['ssim']:.3f}")

    picks = examples[::3][:3]
    cols = [("degraded input", "input"), ("Richardson-Lucy", "richardson_lucy"), ("U-Net", "unet"), ("ground truth", None)]
    fig, ax = plt.subplots(len(picks), 4, figsize=(20, 2.6 * len(picks)))
    ax = np.atleast_2d(ax)
    for r, (outs, tgt) in enumerate(picks):
        for c, (title, key) in enumerate(cols):
            img = tgt if key is None else outs[key]
            ax[r, c].imshow(np.clip(img, 0, 1.2), cmap="gray", vmin=0, vmax=1.1)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(title)
    plt.tight_layout()
    plt.savefig(out_dir / "val_examples.png", dpi=90)


if __name__ == "__main__":
    main()
