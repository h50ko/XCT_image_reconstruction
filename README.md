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
