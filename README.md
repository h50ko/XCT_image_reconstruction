# XCT_image_reconstruction

Machine-learning enhancement of X-ray CT (XCT) slices of a pouch cell: sharpen
the blurred regions of the `bf_*` scans using what the well-resolved regions
(and the sharp reference scans `at_*`) look like.

## Data

| File | Role | Size (px) | Notes |
|------|------|-----------|-------|
| `data/raw/bf_T0.png`, `bf_T0_2.png`, `bf_T0_3.png` | blurred inputs | 468 × 313 | Full circular field of view; coloured viewer border lines are cropped by preprocessing |
| `data/raw/at_T0_ws.png`, `at_T2_ws.png` | sharp references | ~435 × 140 | Cropped to the cell, with a 5 mm scale bar; **not** pixel-aligned with the `bf_*` images |
| `data/raw/reports/<report>/*.png` | all images embedded in the four `Report-2026-*.odt` files | various | Viewer screenshots (XY, YZ, zoomed/full view) with crosshair overlays |
| `data/processed/targets/*.png` | **32 extra sharp targets** | ~452 × 100 | XY zoomed slices from the reports, rescaled to the bf_ pixel scale, cropped to the cell, overlays inpainted; `*_valid.png` marks real (non-overlay) pixels |

Extract targets from report documents (.odt or .docx) with
`python src/extract_targets.py path/to/Report-*.odt`. YZ slices, full-view slices
(cell too small to resolve layers) and blank images are skipped automatically.
After rescaling, the electrode layer pitch in the targets is 2.61–2.72 px, matching
the bf_ scans (~2.6 px).

The pixel scale is similar in both sets (the cell is about 430 px wide in each), so
learned texture statistics transfer between them directly.

### What the blur looks like

* The rounded, wound ends of the cell show well-resolved concentric layers.
* In the long flat section, the layers are smeared **horizontally** (streaky), and
  the top and bottom boundaries of the cell are diffuse. This is typical of a flat,
  high-aspect-ratio sample, where rays along the long axis are strongly attenuated.

## Project layout

```
data/raw/         original uploads (tracked in git)
data/processed/   cropped grayscale images + cell masks (generated, git-ignored)
src/              Python code
results/          output images and figures
```

## Getting started

```bash
pip install -r requirements.txt
python src/preprocess.py      # writes data/processed/
```

Tested with Python 3.13 on CPU (no GPU needed at this image size).

## Git basics for this repo

```bash
git pull                      # get the latest changes from GitHub
git add <files>               # stage your changes
git commit -m "message"       # save a snapshot locally
git push                      # upload the snapshot to GitHub
```

Work for this project happens on the branch `claude/eager-fermat-817d9z`. When you
are happy with the results, open a Pull Request on GitHub to merge it into `main`.

## Approach: synthetic-pair U-Net

1. `src/degrade.py` blurs sharp targets the way the bf_ scans are blurred: a narrow
   core + wide, mostly vertical halo PSF (stronger in the middle of the cell than at
   the ends), contrast loss, haze, horizontal streaks and noise. All parameters are
   randomised so the model learns a range of blur strengths.
2. `src/train.py` trains a 1.9M-parameter residual U-Net (`src/model.py`) on
   (degraded, clean) patch pairs from 31 sharp targets. The 3 slices of
   `Report-2026-03-27_r2` are held out for validation. Overlay pixels are masked
   out of the loss. 6000 iterations take about 37 min on 4 CPU cores.
3. `src/evaluate.py` scores the held-out slices; `src/apply.py` restores the
   bf_ scans and writes comparison figures.

```bash
python src/train.py --iters 6000
python src/evaluate.py
python src/apply.py
```

## Results

Held-out validation (3 sharp slices from an unseen report, each degraded with 3
fixed random seeds; higher is better):

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| Degraded input | 20.44 | 0.533 |
| Richardson–Lucy deconvolution | 19.82 | 0.583 |
| **U-Net** | **23.73** | **0.755** |

Outputs in `results/unet/`:
* `bf_T0*_unet.png`: restored scans (same size as `data/processed/bf_T0*.png`)
* `bf_T0*_rl.png`: Richardson–Lucy baseline
* `bf_T0*_compare.png`: input / RL / U-Net, full view + left end, middle, right end
* `val_examples.png`, `val_metrics.json`, `train_log.json`
* `checkpoints/best.pt`: trained weights

### What to trust

* **Middle section:** noise and haze are removed, and the layers become distinct
  lines. The top and bottom boundaries are sharper.
* **Caveat, invented detail:** where the input has lost the layer structure (the diffuse band at the
  top and bottom of the flat section), the network draws layers that are
  *plausible* given the training targets, not measured. Do not use them for
  quantitative layer counting or thickness measurements without checking against
  a sharp scan.
* **Known regressions:** features absent from the training targets are
  under-restored. The bright tab/pocket in bf_T0_2/3 loses contrast, and the small
  concentric ring at the right end is softened. The rounded ends were already
  sharp and change little.
* Validation uses *synthetic* blur. Real bf_ blur differs somewhat, so the gain on
  real scans is smaller than the table suggests. A sharp and a blurry scan of the
  same slice would allow a true measurement.
