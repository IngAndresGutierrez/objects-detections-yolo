"""Genera un mosaico por particion con las cajas ground truth para revisar las etiquetas.

Uso:
  uv run python src/review_labels.py --data dataset --out outputs/label_review
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

TILE = 320
COLS = 5


def draw_gt(img_path: Path, label_path: Path) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for line in label_path.read_text().split("\n"):
        if not line.strip():
            continue
        _, cx, cy, bw, bh = map(float, line.split())
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
    draw.text((5, 5), img_path.stem, fill=(255, 255, 0))
    img.thumbnail((TILE, TILE))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--out", type=Path, default=Path("outputs/label_review"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        imgs = sorted((args.data / "images" / split).glob("*.jpg"))
        rows = (len(imgs) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * TILE, rows * TILE), (30, 30, 30))
        for i, p in enumerate(imgs):
            tile = draw_gt(p, args.data / "labels" / split / f"{p.stem}.txt")
            sheet.paste(tile, ((i % COLS) * TILE, (i // COLS) * TILE))
        sheet.save(args.out / f"{split}.jpg", quality=85)
        print(f"{split}: {len(imgs)} imagenes -> {args.out / f'{split}.jpg'}")


if __name__ == "__main__":
    main()
