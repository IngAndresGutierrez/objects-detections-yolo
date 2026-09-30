"""Prepara un dataset YOLO de una sola clase (car) a partir de PASCAL VOC 2007.

Pasos:
  1. Lee las anotaciones XML de VOC y se queda con imagenes que contienen coches.
  2. Descarta imagenes con coches marcados como `difficult` (etiqueta ambigua).
  3. Selecciona N imagenes estratificando por numero de coches y tamano del coche mayor.
  4. Elimina casi-duplicados (dHash) antes de repartir en train/val/test.
  5. Escribe images/, labels/ (formato YOLO normalizado, clase 0), data.yaml y
     un split.csv con la lista de imagenes de cada particion.

Uso:
  uv run python src/prepare_data.py --voc data/raw/VOCdevkit/VOC2007 --out dataset
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image

CLASS_NAME = "car"


@dataclass
class Sample:
    image_id: str
    width: int
    height: int
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)  # xmin, ymin, xmax, ymax

    @property
    def n_cars(self) -> int:
        return len(self.boxes)

    @property
    def max_area_ratio(self) -> float:
        return max((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in self.boxes) / (self.width * self.height)

    @property
    def count_bin(self) -> str:
        return "1" if self.n_cars == 1 else "2-3" if self.n_cars <= 3 else "4+"

    @property
    def size_bin(self) -> str:
        r = self.max_area_ratio
        return "small" if r < 0.05 else "medium" if r < 0.25 else "large"


def parse_voc(voc_dir: Path) -> list[Sample]:
    samples = []
    for xml_path in sorted((voc_dir / "Annotations").glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        w, h = int(root.findtext("size/width", "0")), int(root.findtext("size/height", "0"))
        cars, has_difficult = [], False
        for obj in root.iter("object"):
            if obj.findtext("name") != CLASS_NAME:
                continue
            if int(obj.findtext("difficult", "0")) == 1:
                has_difficult = True
                break
            x1, y1, x2, y2 = (float(obj.findtext(f"bndbox/{k}", "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
            cars.append((x1, y1, x2, y2))
        if cars and not has_difficult:
            samples.append(Sample(xml_path.stem, w, h, cars))
    return samples


def dhash(path: Path, size: int = 8) -> int:
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(img.tobytes())
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (px[row * (size + 1) + col] > px[row * (size + 1) + col + 1])
    return bits


def select_stratified(samples: list[Sample], n: int, rng: random.Random) -> list[Sample]:
    """Reparte n imagenes lo mas uniformemente posible entre los estratos (n_coches x tamano)."""
    strata: dict[tuple[str, str], list[Sample]] = {}
    for s in samples:
        strata.setdefault((s.count_bin, s.size_bin), []).append(s)
    for group in strata.values():
        rng.shuffle(group)
    selected: list[Sample] = []
    keys = sorted(strata)
    while len(selected) < n and any(strata.values()):
        for k in keys:
            if strata[k] and len(selected) < n:
                selected.append(strata[k].pop())
    return selected


def remove_near_duplicates(samples: list[Sample], img_dir: Path, max_dist: int) -> tuple[list[Sample], list[str]]:
    kept, hashes, dropped = [], [], []
    for s in samples:
        h = dhash(img_dir / f"{s.image_id}.jpg")
        if any(bin(h ^ other).count("1") <= max_dist for other in hashes):
            dropped.append(s.image_id)
            continue
        kept.append(s)
        hashes.append(h)
    return kept, dropped


def to_yolo_lines(s: Sample) -> list[str]:
    lines = []
    for x1, y1, x2, y2 in s.boxes:
        # VOC usa coordenadas 1-based en pixeles
        cx = ((x1 + x2) / 2 - 1) / s.width
        cy = ((y1 + y2) / 2 - 1) / s.height
        bw = (x2 - x1) / s.width
        bh = (y2 - y1) / s.height
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def split_stratified(samples: list[Sample], sizes: dict[str, int], rng: random.Random) -> dict[str, list[Sample]]:
    """Ordena por estrato e intercala para que cada particion reciba una mezcla de escenas."""
    by_stratum: dict[tuple[str, str], list[Sample]] = {}
    for s in samples:
        by_stratum.setdefault((s.count_bin, s.size_bin), []).append(s)
    interleaved: list[Sample] = []
    groups = [by_stratum[k] for k in sorted(by_stratum)]
    rng.shuffle(groups)
    for g in groups:
        rng.shuffle(g)
    while any(groups):
        for g in groups:
            if g:
                interleaved.append(g.pop())
    # cada imagen va a la particion mas alejada de su cuota proporcional, asi los
    # estratos se reparten en la misma proporcion 35/10/5 (empates: test > val > train)
    order = ["test", "val", "train"]
    splits: dict[str, list[Sample]] = {k: [] for k in order}
    for i, s in enumerate(interleaved, start=1):
        total = sum(sizes.values())
        k = max((k for k in order if len(splits[k]) < sizes[k]),
                key=lambda k: sizes[k] * i / total - len(splits[k]))
        splits[k].append(s)
    return splits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voc", type=Path, default=Path("data/raw/VOCdevkit/VOC2007"))
    ap.add_argument("--out", type=Path, default=Path("dataset"))
    ap.add_argument("--train", type=int, default=35)
    ap.add_argument("--val", type=int, default=10)
    ap.add_argument("--test", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dup-dist", type=int, default=6, help="distancia Hamming maxima (dHash 64 bits) para considerar casi-duplicado")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sizes = {"train": args.train, "val": args.val, "test": args.test}
    total = sum(sizes.values())
    img_dir = args.voc / "JPEGImages"

    samples = parse_voc(args.voc)
    print(f"Imagenes VOC con coches (sin 'difficult'): {len(samples)}")

    # sobre-muestreamos para poder descartar duplicados sin quedarnos cortos
    candidates = select_stratified(samples, total * 2, rng)
    candidates, dropped = remove_near_duplicates(candidates, img_dir, args.dup_dist)
    if dropped:
        print(f"Casi-duplicados descartados: {dropped}")
    selected = select_stratified(candidates, total, rng)
    assert len(selected) == total, f"Solo hay {len(selected)} imagenes validas, se pedian {total}"

    splits = split_stratified(selected, sizes, rng)

    if args.out.exists():
        shutil.rmtree(args.out)
    rows = []
    for split, items in splits.items():
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)
        for s in sorted(items, key=lambda x: x.image_id):
            shutil.copy(img_dir / f"{s.image_id}.jpg", args.out / "images" / split / f"{s.image_id}.jpg")
            (args.out / "labels" / split / f"{s.image_id}.txt").write_text("\n".join(to_yolo_lines(s)) + "\n")
            rows.append({"split": split, "image_id": s.image_id, "width": s.width, "height": s.height,
                         "n_cars": s.n_cars, "count_bin": s.count_bin, "size_bin": s.size_bin,
                         "max_area_ratio": round(s.max_area_ratio, 4)})

    data_yaml = {"path": str(args.out.resolve()), "train": "images/train", "val": "images/val",
                 "test": "images/test", "names": {0: CLASS_NAME}}
    (args.out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    with open(args.out / "split.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for split, items in splits.items():
        n_boxes = sum(s.n_cars for s in items)
        mix = {}
        for s in items:
            mix[(s.count_bin, s.size_bin)] = mix.get((s.count_bin, s.size_bin), 0) + 1
        print(f"{split:5s}: {len(items):2d} imagenes, {n_boxes:3d} coches | estratos {dict(sorted(mix.items()))}")
    print(f"Dataset escrito en {args.out.resolve()}")


if __name__ == "__main__":
    main()
