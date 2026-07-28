"""Extract Zhang's diagnosis-related figures/tables from the paper PDF as PNGs.

These are REAL renders of regions of `docs/BN-NTSB RESS 2021.pdf` (Zhang & Mahadevan,
RESS 2021), used for side-by-side comparison in the diagnosis deck. Nothing here is
fabricated — each PNG is a clipped, high-DPI render of the source page region.

Run with the framework interpreter:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 docs/extract_zhang_figures.py
"""
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

HERE = Path(__file__).resolve().parent
PDF = HERE / "BN-NTSB RESS 2021.pdf"
OUT = HERE / "figures" / "zhang"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 200
ZOOM = DPI / 72.0
MAT = fitz.Matrix(ZOOM, ZOOM)

doc = fitz.open(str(PDF))


def render_clip(page_index, rect_pts, name):
    """Render a clip (in unrotated PDF points) of a non-rotated page."""
    page = doc[page_index]
    clip = fitz.Rect(*rect_pts)
    pix = page.get_pixmap(matrix=MAT, clip=clip)
    path = OUT / name
    pix.save(str(path))
    return path


def render_full_then_crop(page_index, px_box, name):
    """Render the full (auto-rotated) page, then PIL-crop in pixel space.

    Used for rotated/landscape pages where clip rects are awkward.
    """
    page = doc[page_index]
    pix = page.get_pixmap(matrix=MAT)
    tmp = OUT / ("_tmp_" + name)
    pix.save(str(tmp))
    img = Image.open(str(tmp))
    img.crop(px_box).save(str(OUT / name))
    tmp.unlink()
    return OUT / name


results = []

# (a) Table 7 — P(cause | fire). Page index 11 is a rotated (90 deg) full-page landscape
#     table. Crop to the measured content bounding box of the 200-DPI render (2205x1654),
#     trimmed slightly to drop the rotated journal header/footer at the page margins.
t7_box = (84, 128, 2096, 1546)
results.append(("zhang_table7.png", render_full_then_crop(11, t7_box, "zhang_table7.png"),
                "Table 7: contributory factors to fire & P(cause|fire)"))

# (b) Table 4 — CPT of the Fire node x3 (brake wear + wiring overheat -> fire).
results.append(("zhang_table4.png", render_clip(4, (34, 293, 515, 372), "zhang_table4.png"),
                "Table 4: pedagogical CPT P(fire | brake wear, wiring overheat)"))

# (c) Fig. 2 — simple Bayesian network: fire node (x3) with its parent causes
#     (x1 landing-gear brake wear, x2 wiring overheat) and child x4 aircraft damage.
results.append(("zhang_bn_fire.png", render_clip(4, (110, 52, 488, 213), "zhang_bn_fire.png"),
                "Fig. 2: BN with Fire node + parent causes"))

# (d) Prior probability estimation (Section 4.2): Eq. (6) and the explicit fire example
#     P(fire) = 102 / 184,517,128 ~ 5.52e-7.
results.append(("zhang_prior.png", render_clip(5, (303, 505, 563, 748), "zhang_prior.png"),
                "Section 4.2 prior: P(fire)=102/184,517,128"))

# (d') Fig. 8 — Beta-CDF conditional-probability method (Section 4.3 / 5.1).
results.append(("zhang_betacdf.png", render_clip(10, (110, 54, 500, 316), "zhang_betacdf.png"),
                "Fig. 8: Beta-CDF conditional-probability estimator"))

print("Extracted figures:")
for name, path, desc in results:
    size = path.stat().st_size
    with Image.open(str(path)) as im:
        w, h = im.size
    flag = "OK" if size > 3000 else "** TOO SMALL **"
    print(f"  {name:22s} {w}x{h}px {size/1024:6.1f} KB  {flag}  <- {desc}")
