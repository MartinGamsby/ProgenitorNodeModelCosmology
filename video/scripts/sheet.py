"""Tile review stills into one contact sheet: python scripts/sheet.py out/sheet.jpg a.jpg b.jpg ..."""
import sys
from PIL import Image, ImageDraw
out, files = sys.argv[1], sys.argv[2:]
ims = [Image.open(f) for f in files]
w, h = ims[0].size
cols = 2
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (w * cols, h * rows), "black")
d = ImageDraw.Draw(sheet)
for k, im in enumerate(ims):
    x, y = (k % cols) * w, (k // cols) * h
    sheet.paste(im, (x, y))
    d.text((x + 8, y + 8), files[k].split("/")[-1].split("\\")[-1], fill="yellow")
sheet.save(out, quality=82)
