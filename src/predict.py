"""Inferencia del YOLO elegido sobre el test y galeria original | prediccion.

Lee models/selection.json (imgsz y umbral fijados con validacion) y guarda en outputs/yolo_scratch/:
  - predictions.json: TODAS las cajas con conf >= 0.001 (pixeles xyxy + confianza) y el ground
    truth por imagen; el umbral elegido va en meta.conf_threshold (necesario para calcular AP)
  - labels/<id>.txt: predicciones con conf >= umbral en formato YOLO (clase cx cy w h conf)
  - gallery/<id>.jpg: original sin marcas a la izquierda, predicciones (rojo, con confianza)
    y ground truth (verde) a la derecha
  - timing.json: tiempo medio de inferencia por imagen

Uso:
  uv run python src/predict.py --split test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
PRED_COLOR, GT_COLOR = (230, 30, 30), (0, 200, 0)
MIN_CONF = 0.001


def read_gt(label_path: Path, w: int, h: int) -> list[list[float]]:
    boxes = []
    for line in label_path.read_text().split("\n"):
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split())
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def side_by_side(img: Image.Image, preds: list[dict], gt: list[list[float]], show_conf: bool = True) -> Image.Image:
    right = img.copy()
    draw = ImageDraw.Draw(right)
    for box in gt:
        draw.rectangle(box, outline=GT_COLOR, width=2)
    for p in preds:
        x1, y1, x2, y2 = p["xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=PRED_COLOR, width=3)
        draw.text((x1 + 3, max(0, y1 - 12)), f"car {p['conf']:.2f}" if show_conf else "car", fill=PRED_COLOR)
    if not preds:
        draw.text((8, 8), "SIN DETECCIONES", fill=PRED_COLOR)
    draw.text((8, img.height - 16), f"pred={len(preds)}  gt={len(gt)}  (rojo=pred, verde=GT)", fill=(255, 255, 0))
    canvas = Image.new("RGB", (img.width * 2 + 10, img.height), (255, 255, 255))
    canvas.paste(img, (0, 0))
    canvas.paste(right, (img.width + 10, 0))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", type=Path, default=ROOT / "dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "yolo_scratch")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    sel = json.loads((ROOT / "models" / "selection.json").read_text())
    model = YOLO(ROOT / sel["weights"])
    for sub in ("gallery", "labels"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    images = sorted((args.data / "images" / args.split).glob("*.jpg"))
    model.predict(images[0], imgsz=sel["imgsz"], device=args.device, verbose=False)  # calentamiento

    results, times = {}, []
    for path in images:
        img = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        r = model.predict(path, imgsz=sel["imgsz"], conf=MIN_CONF, device=args.device, verbose=False)[0]
        times.append(time.perf_counter() - t0)

        preds = [{"xyxy": [round(v, 1) for v in b.xyxy[0].tolist()], "conf": round(float(b.conf), 4)} for b in r.boxes]
        kept = [p for p in preds if p["conf"] >= sel["conf_threshold"]]
        gt = read_gt(args.data / "labels" / args.split / f"{path.stem}.txt", img.width, img.height)
        results[path.stem] = {"width": img.width, "height": img.height, "predictions": preds,
                              "ground_truth": [[round(v, 1) for v in b] for b in gt]}

        yolo_lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {float(c):.4f}"
                      for (cx, cy, w, h), c in zip(r.boxes.xywhn.tolist(), r.boxes.conf.tolist())
                      if c >= sel["conf_threshold"]]
        (args.out / "labels" / f"{path.stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        side_by_side(img, kept, gt).save(args.out / "gallery" / f"{path.stem}.jpg", quality=90)
        print(f"{path.stem}: {len(kept)} detecciones / {len(gt)} coches reales")

    meta = {"model": sel["weights"], "imgsz": sel["imgsz"], "conf_threshold": sel["conf_threshold"],
            "min_conf_saved": MIN_CONF, "method": "YOLO26n desde cero", "split": args.split, "device": args.device, "n_images": len(images)}
    (args.out / "predictions.json").write_text(json.dumps({"meta": meta, "images": results}, indent=2))
    (args.out / "timing.json").write_text(json.dumps(
        {**meta, "mean_ms_per_image": round(1000 * sum(times) / len(times), 1),
         "per_image_ms": {p.stem: round(1000 * t, 1) for p, t in zip(images, times)}}, indent=2))
    print(f"Media: {1000 * sum(times) / len(times):.1f} ms/imagen -> {args.out}")


if __name__ == "__main__":
    main()
