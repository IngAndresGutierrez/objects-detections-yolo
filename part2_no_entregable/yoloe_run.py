"""(NO ENTREGABLE) Detector preentrenado YOLOE-26 con clases por texto, sin entrenar.

1. En VALIDACION: prueba varias listas de clases (prompts), guarda todas las cajas (conf >= 0.001)
   y elige la variante con mayor AP50; el umbral de confianza es el que maximiza F1 en validacion.
2. En TEST: ejecuta la variante y el umbral elegidos (fijos) y guarda predicciones, tiempos y galeria.
Todas las clases del prompt se consideran "car" (VOC incluye furgonetas, pickups y todoterrenos).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import Image
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import best_f1_threshold, evaluate, load_split_info  # noqa: E402
from predict import read_gt, side_by_side  # noqa: E402

WEIGHTS = "yoloe-26s-seg.pt"
IMGSZ = 640
MIN_CONF = 0.001
PROMPTS = {
    "car": ["car"],
    "car+van+pickup": ["car", "van", "pickup truck"],
    "vehicle": ["vehicle"],
    "automobile": ["automobile"],
}
OUT = ROOT / "outputs" / "yoloe"


def run(model: YOLOE, split: str, method: str, conf_thr: float) -> tuple[dict, list[float]]:
    images = sorted((ROOT / "dataset" / "images" / split).glob("*.jpg"))
    model.predict(images[0], imgsz=IMGSZ, device="cpu", verbose=False)  # calentamiento
    out, times = {}, []
    for path in images:
        img = Image.open(path)
        t0 = time.perf_counter()
        r = model.predict(path, imgsz=IMGSZ, conf=MIN_CONF, device="cpu", verbose=False)[0]
        times.append(time.perf_counter() - t0)
        preds = [{"xyxy": [round(v, 1) for v in b.xyxy[0].tolist()], "conf": round(float(b.conf), 4),
                  "prompt_class": r.names[int(b.cls)]} for b in r.boxes]
        gt = read_gt(ROOT / "dataset" / "labels" / split / f"{path.stem}.txt", img.width, img.height)
        out[path.stem] = {"width": img.width, "height": img.height, "predictions": preds,
                          "ground_truth": [[round(v, 1) for v in b] for b in gt]}
    meta = {"method": method, "conf_threshold": conf_thr, "has_scores": True, "min_conf_saved": MIN_CONF,
            "weights": WEIGHTS, "imgsz": IMGSZ, "split": split, "device": "cpu"}
    return {"meta": meta, "images": out}, times


def main() -> None:
    split_info = load_split_info(ROOT / "dataset" / "split.csv")
    (OUT / "val_prompts").mkdir(parents=True, exist_ok=True)

    val_rows = []
    for name, classes in PROMPTS.items():
        model = YOLOE(WEIGHTS)
        model.set_classes(classes)
        data, _ = run(model, "val", f"YOLOE-26s [{name}]", 0.0)
        thr, f1 = best_f1_threshold(data)
        data["meta"]["conf_threshold"] = thr
        res = evaluate(data, split_info)
        (OUT / "val_prompts" / f"{name}.json").write_text(json.dumps(data, indent=1))
        row = {"prompt": name, "classes": classes, "val_AP50": res["AP50"], "val_AP50-95": res["AP50-95"],
               "best_conf": round(thr, 4), "val_F1": round(f1, 4), **res["operating_point"]}
        val_rows.append(row)
        print(json.dumps(row))

    best = max(val_rows, key=lambda r: (r["val_AP50"], r["val_AP50-95"]))
    (OUT / "prompt_selection_val.json").write_text(json.dumps({"criterion": "max val AP50, desempate AP50-95",
                                                               "chosen": best, "all": val_rows}, indent=2))
    print(f"\nElegido en validacion: {best['prompt']} (conf={best['best_conf']})")

    model = YOLOE(WEIGHTS)
    model.set_classes(best["classes"])
    data, times = run(model, "test", f"YOLOE-26s [{best['prompt']}]", best["best_conf"])
    (OUT / "predictions.json").write_text(json.dumps(data, indent=2))
    (OUT / "timing.json").write_text(json.dumps({"mean_ms_per_image": round(1000 * sum(times) / len(times), 1),
                                                 "device": "cpu", "imgsz": IMGSZ}, indent=2))
    (OUT / "gallery").mkdir(exist_ok=True)
    for img_id, v in data["images"].items():
        kept = [p for p in v["predictions"] if p["conf"] >= best["best_conf"]]
        img = Image.open(ROOT / "dataset" / "images" / "test" / f"{img_id}.jpg").convert("RGB")
        side_by_side(img, kept, v["ground_truth"]).save(OUT / "gallery" / f"{img_id}.jpg", quality=90)
        print(f"{img_id}: {len(kept)} detecciones / {len(v['ground_truth'])} coches reales")


if __name__ == "__main__":
    main()
