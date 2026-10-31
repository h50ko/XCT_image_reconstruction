"""Train the restoration U-Net on synthetic (degraded at_*, clean at_*) pairs.

Train set: all of at_T0_ws + the left 75 % of at_T2_ws.
Validation: the right 25 % of at_T2_ws (never seen in training), degraded
with a fixed set of random seeds so scores are comparable across runs.

Usage:  python src/train.py [--iters 6000] [--patch 96] [--batch 16]
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import PROCESSED, RESULTS, load_gray, norm_params, normalise
from degrade import degrade
from model import UNet

VAL_FRAC = 0.25


def load_targets():
    out = {}
    for name in ["at_T0_ws", "at_T2_ws"]:
        img = load_gray(PROCESSED / f"{name}.png")
        lo, sc = norm_params(img)
        out[name] = normalise(img, lo, sc)
    a2 = out["at_T2_ws"]
    cut = int(a2.shape[1] * (1 - VAL_FRAC))
    train = [out["at_T0_ws"], a2[:, :cut]]
    val = a2[:, cut - 16:]  # small overlap is fine; ends are never trained on
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
            for img in train:
                x = img.copy()
                if rng.random() < 0.5:
                    x = x[:, ::-1]
                if rng.random() < 0.5:
                    x = x[::-1, :]
                x = np.ascontiguousarray(x)
                pool.append((make_input(x, rng), x))
        xs, ys = [], []
        for _ in range(bsz):
            inp, tgt = pool[rng.integers(len(pool))]
            h, w = tgt.shape
            ph = min(patch, h)
            r, c = rng.integers(0, h - ph + 1), rng.integers(0, w - patch + 1)
            xs.append(inp[r:r + ph, c:c + patch])
            ys.append(tgt[r:r + ph, c:c + patch])
        step += 1
        yield (torch.from_numpy(np.stack(xs))[:, None], torch.from_numpy(np.stack(ys))[:, None])


def grad_loss(p, t):
    dx = lambda z: z[..., :, 1:] - z[..., :, :-1]
    dy = lambda z: z[..., 1:, :] - z[..., :-1, :]
    return F.l1_loss(dx(p), dx(t)) + F.l1_loss(dy(p), dy(t))


def psnr(p, t):
    return float(10 * np.log10(1.0 / np.mean((np.clip(p, 0, 1.2) - t) ** 2)))


def make_val_set(val, n=8):
    return [(make_input(val, np.random.default_rng(1000 + i)), val) for i in range(n)]


@torch.no_grad()
def evaluate(net, vset):
    net.eval()
    s_in, s_out = [], []
    for inp, tgt in vset:
        out = net(torch.from_numpy(inp)[None, None]).numpy()[0, 0]
        s_in.append(psnr(inp, tgt))
        s_out.append(psnr(out, tgt))
    net.train()
    return float(np.mean(s_in)), float(np.mean(s_out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, val = load_targets()
    vset = make_val_set(val)
    net = UNet()
    print(f"params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.iters, pct_start=0.05)

    out_dir = RESULTS / "unet"
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log, best = [], -1.0
    t0 = time.time()
    for it, (x, y) in enumerate(batch_iter(train, rng, args.patch, args.batch), 1):
        p = net(x)
        loss = F.l1_loss(p, y) + 0.5 * grad_loss(p, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 250 == 0 or it == args.iters:
            v_in, v_out = evaluate(net, vset)
            log.append(dict(iter=it, loss=float(loss), val_psnr_input=v_in, val_psnr_output=v_out))
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
