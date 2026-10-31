"""Train the restoration U-Net on synthetic (degraded target, clean target) pairs.

Targets: at_T0_ws, at_T2_ws and the zoomed XY slices extracted from the
reports (data/processed/targets, see extract_targets.py).
Validation: every slice from one held-out report (default Report-2026-03-27_r2),
degraded with fixed random seeds so scores are comparable across runs.
Overlay pixels (crosshairs, scale bar) are excluded from the loss.

Usage:  python src/train.py [--iters 6000] [--patch 80] [--batch 16]
"""
import argparse
import json
import time

import numpy as np
import torch
from PIL import Image

from common import PROCESSED, RESULTS, load_gray, norm_params, normalise
from degrade import degrade
from model import UNet


def load_targets(holdout: str):
    """Return lists of (clean_normalised, valid_mask) for train and val."""
    items = []
    for name in ["at_T0_ws", "at_T2_ws"]:
        img = load_gray(PROCESSED / f"{name}.png")
        valid = np.ones(img.shape, bool)
        valid[112:, 370:] = False  # scale bar (filled by preprocess.py)
        items.append((name, img, valid))
    for p in sorted((PROCESSED / "targets").glob("*.png")):
        if p.stem.endswith("_valid"):
            continue
        valid = np.array(Image.open(p.with_name(p.stem + "_valid.png"))) > 0
        items.append((p.stem, load_gray(p), valid))
    train, val = [], []
    for name, img, valid in items:
        lo, sc = norm_params(img, valid)
        pair = (normalise(img, lo, sc).astype(np.float32), valid)
        (val if name.startswith(holdout) else train).append(pair)
    return train, val


def make_input(clean: np.ndarray, rng) -> np.ndarray:
    """Degrade, then normalise exactly the way the real bf_ scans are."""
    d = degrade(clean, rng)
    lo, sc = norm_params(d * 255)
    return normalise(d * 255, lo, sc).astype(np.float32)


def batch_iter(train, rng, patch, bsz, refresh=4):
    """Yield (input, target) patch batches.  The whole image is re-degraded
    every few batches (images are tiny, so this is cheap), which keeps the
    spatially varying blur realistic instead of per-patch."""
    pool = []
    step = 0
    while True:
        if step % refresh == 0:
            pool = []
            for k in rng.choice(len(train), size=min(8, len(train)), replace=False):
                x, v = train[k]
                if rng.random() < 0.5:
                    x, v = x[:, ::-1], v[:, ::-1]
                if rng.random() < 0.5:
                    x, v = x[::-1, :], v[::-1, :]
                x, v = np.ascontiguousarray(x), np.ascontiguousarray(v)
                pool.append((make_input(x, rng), x, v))
        xs, ys, vs = [], [], []
        for _ in range(bsz):
            inp, tgt, v = pool[rng.integers(len(pool))]
            h, w = tgt.shape
            r, c = rng.integers(0, h - patch + 1), rng.integers(0, w - patch + 1)
            xs.append(inp[r:r + patch, c:c + patch])
            ys.append(tgt[r:r + patch, c:c + patch])
            vs.append(v[r:r + patch, c:c + patch])
        step += 1
        t = lambda a: torch.from_numpy(np.stack(a).astype(np.float32))[:, None]
        yield t(xs), t(ys), t(vs)


def masked_l1(a, b, m):
    return (torch.abs(a - b) * m).sum() / m.sum().clamp(min=1)


def loss_fn(p, t, m):
    dx = lambda z: z[..., :, 1:] - z[..., :, :-1]
    dy = lambda z: z[..., 1:, :] - z[..., :-1, :]
    mx, my = m[..., :, 1:] * m[..., :, :-1], m[..., 1:, :] * m[..., :-1, :]
    return masked_l1(p, t, m) + 0.5 * (masked_l1(dx(p), dx(t), mx) + masked_l1(dy(p), dy(t), my))


def psnr(p, t, m):
    return float(10 * np.log10(1.0 / np.mean(((np.clip(p, 0, 1.2) - t) ** 2)[m])))


def make_val_set(val, reps=3):
    return [(make_input(x, np.random.default_rng(1000 + 10 * j + i)), x, v)
            for j, (x, v) in enumerate(val) for i in range(reps)]


@torch.no_grad()
def evaluate(net, vset):
    net.eval()
    s_in, s_out = [], []
    for inp, tgt, v in vset:
        out = net(torch.from_numpy(inp)[None, None]).numpy()[0, 0]
        s_in.append(psnr(inp, tgt, v))
        s_out.append(psnr(out, tgt, v))
    net.train()
    return float(np.mean(s_in)), float(np.mean(s_out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--patch", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", default="Report-2026-03-27_r2")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, val = load_targets(args.holdout)
    vset = make_val_set(val)
    print(f"train images: {len(train)}, val images: {len(val)} ({args.holdout})")
    net = UNet()
    print(f"params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.iters, pct_start=0.05)

    out_dir = RESULTS / "unet"
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log, best = [], -1.0
    t0 = time.time()
    for it, (x, y, m) in enumerate(batch_iter(train, rng, args.patch, args.batch), 1):
        p = net(x)
        loss = loss_fn(p, y, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 250 == 0 or it == args.iters:
            v_in, v_out = evaluate(net, vset)
            log.append(dict(iter=it, loss=loss.item(), val_psnr_input=v_in, val_psnr_output=v_out))
            print(f"it {it:5d}  loss {loss.item():.4f}  val PSNR in {v_in:.2f} -> out {v_out:.2f} dB"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
            if v_out > best:
                best = v_out
                torch.save(net.state_dict(), out_dir / "checkpoints/best.pt")
        if it >= args.iters:
            break
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"best val PSNR {best:.2f} dB")


if __name__ == "__main__":
    main()
