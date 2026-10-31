"""Extract sharp target slices from report documents (.odt / .docx).

Reports are zip archives; embedded images live in Pictures/ (LibreOffice) or
media/ / word/media/ (Word).  For every embedded image this script:

1. saves the original to data/raw/reports/<report>/<name>.png,
2. classifies it:  XY slice (cell wider than tall) -> used as a target;
   YZ slice, blank or unrecognised -> skipped,
3. builds a mask of viewer overlays (coloured crosshairs, scale bar),
4. rescales so the cell width matches the bf_ scans (~425 px, which puts the
   electrode layer pitch near the ~2.6 px seen in bf_),
5. crops to the cell (+ margin, which also removes the viewer text), fills
   overlay pixels by inpainting, and writes
   data/processed/targets/<report>__<name>.png  and  ..._valid.png
   (valid = 255 where the pixel is real image data; used to mask the loss).

Usage:  python src/extract_targets.py REPORT1.odt [REPORT2.odt ...]
"""
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from common import PROCESSED, ROOT, cell_mask

RAW_OUT = ROOT / "data/raw/reports"
TGT_OUT = PROCESSED / "targets"
TARGET_CELL_WIDTH = 425      # px, cell width in the bf_ scans
MIN_NATIVE_WIDTH = 500       # skip views where the cell is too small (layers unresolved)
MARGIN = 14                  # px around the cell after rescaling


def report_name(path: Path) -> str:
    m = re.search(r"Report-[\d-]+(?:_r\d+)?", path.stem)
    return m.group(0) if m else path.stem


def embedded_images(path: Path):
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir() or not re.match(r"(Pictures|media|word/media)/", info.filename):
                continue
            data = z.read(info)
            try:
                img = Image.open(io.BytesIO(data))
                img.load()
            except Exception:
                continue
            yield Path(info.filename).stem, data, img


def overlay_mask(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(int)
    coloured = (np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])) > 30
    return ndi.binary_dilation(coloured, iterations=1)


def process(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if gray.max() < 30:
        return None, "blank"
    over = overlay_mask(rgb)
    filled = cv2.inpaint(gray, over.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    m = cell_mask(filled.astype(np.float32))
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None, "no cell"
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    if h > w:
        return None, "YZ view"
    if w < MIN_NATIVE_WIDTH:
        return None, f"cell only {w}px wide (layers unresolved)"
    s = TARGET_CELL_WIDTH / w
    y0, y1 = max(ys.min() - int(MARGIN / s), 0), min(ys.max() + int(MARGIN / s), gray.shape[0])
    x0, x1 = max(xs.min() - int(MARGIN / s), 0), min(xs.max() + int(MARGIN / s), gray.shape[1])
    crop, valid = filled[y0:y1, x0:x1], ~over[y0:y1, x0:x1]
    size = (round(crop.shape[1] * s), round(crop.shape[0] * s))
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    valid = cv2.resize(valid.astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) == 255
    return (crop, valid, s), "ok"


def main(paths):
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    TGT_OUT.mkdir(parents=True, exist_ok=True)
    seen = set()
    for p in map(Path, paths):
        rep = report_name(p)
        (RAW_OUT / rep).mkdir(exist_ok=True)
        for name, data, img in embedded_images(p):
            digest = hashlib.sha1(data).hexdigest()
            short = name[-8:]
            (RAW_OUT / rep / f"{short}.png").write_bytes(data) if img.format == "PNG" else \
                img.save(RAW_OUT / rep / f"{short}.png")
            if digest in seen:
                print(f"{rep}/{short}: duplicate, skipped")
                continue
            seen.add(digest)
            res, why = process(np.array(img.convert("RGB")))
            if res is None:
                print(f"{rep}/{short}: skipped ({why})")
                continue
            crop, valid, s = res
            stem = f"{rep}__{short}"
            Image.fromarray(crop).save(TGT_OUT / f"{stem}.png")
            Image.fromarray(valid.astype(np.uint8) * 255).save(TGT_OUT / f"{stem}_valid.png")
            print(f"{rep}/{short}: target {crop.shape[1]}x{crop.shape[0]} (scale {s:.2f}, "
                  f"{100 * (1 - valid.mean()):.1f}% overlay masked)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
