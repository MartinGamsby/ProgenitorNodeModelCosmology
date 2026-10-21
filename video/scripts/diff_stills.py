"""Compare two folders of stills (e.g. before/after a refactor): python scripts/diff_stills.py out/baseline out/stills
Prints frames whose pixels differ beyond JPEG noise and writes out/diff-sheet.jpg (before | after | diff)."""
import os
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw

a_dir, b_dir = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(os.listdir(a_dir), key=lambda s: int(s.split('.')[0])):
    pb = os.path.join(b_dir, f)
    if not os.path.exists(pb):
        rows.append((999, 999, f, 'missing'))
        continue
    A = np.asarray(Image.open(os.path.join(a_dir, f)).convert('L'), dtype=np.int16)
    B = np.asarray(Image.open(pb).convert('L'), dtype=np.int16)
    d = np.abs(A - B)
    # fraction of pixels that changed visibly (JPEG noise stays well below 24 levels)
    changed = float((d > 24).mean() * 100)
    rows.append((changed, float(d.mean()), f, ''))
bad = [r for r in rows if r[0] > 0.05 or r[3]]
print(f"{len(rows)} frames compared, {len(bad)} differ")
for r in sorted(bad, reverse=True)[:20]:
    print(f"  {r[2]}: {r[0]:.2f}% pixels changed, mean diff {r[1]:.2f} {r[3]}")
top = [r[2] for r in sorted(bad, reverse=True)[:6] if not r[3]]
if top:
    tiles = []
    for f in top:
        A = Image.open(os.path.join(a_dir, f)).convert('RGB'); B = Image.open(os.path.join(b_dir, f)).convert('RGB')
        D = ImageChops.difference(A, B).point(lambda v: min(255, v * 4))
        w, h = A.size
        t = Image.new('RGB', (w * 3, h)); t.paste(A, (0, 0)); t.paste(B, (w, 0)); t.paste(D, (2 * w, 0))
        ImageDraw.Draw(t).text((6, 6), f, fill='yellow'); tiles.append(t)
    W, H = tiles[0].size
    sheet = Image.new('RGB', (W, H * len(tiles)))
    for i, t in enumerate(tiles): sheet.paste(t, (0, i * H))
    sheet.save('out/diff-sheet.jpg', quality=80)
    print('out/diff-sheet.jpg')
