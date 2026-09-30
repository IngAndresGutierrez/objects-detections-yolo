"""Entrena un YOLO desde cero (pesos aleatorios) sobre el dataset de coches.

La arquitectura se construye a partir de un .yaml (solo la definicion de capas).
Un .pt contendria pesos ya entrenados; aqui se rechaza explicitamente.

Cada ejecucion guarda en runs/train/<name>/:
  - weights/best.pt y last.pt, results.csv y las curvas de Ultralytics
  - run_info.json con parametros, tiempo, hardware y metricas de validacion

Uso:
  uv run python src/train.py --name img416 --imgsz 416
  uv run python src/train.py --name img640 --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def cpu_name() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="nombre del experimento (carpeta en runs/train)")
    ap.add_argument("--model", default="yolo26n.yaml", help="definicion de arquitectura (.yaml)")
    ap.add_argument("--data", type=Path, default=ROOT / "dataset" / "data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=0, help="0 = sin early stopping, para ver la curva completa")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    if not args.model.endswith(".yaml"):
        raise SystemExit(f"'{args.model}' no es un .yaml: para entrenar desde cero no se cargan pesos (.pt)")

    model = YOLO(args.model)  # construye la red con inicializacion aleatoria
    n_params = sum(p.numel() for p in model.model.parameters())

    start = time.perf_counter()
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        workers=args.workers,
        pretrained=False,
        project=str(ROOT / "runs" / "train"),
        name=args.name,
        exist_ok=True,
        plots=True,
    )
    elapsed = time.perf_counter() - start

    save_dir = Path(model.trainer.save_dir)
    best = YOLO(save_dir / "weights" / "best.pt")
    val = best.val(data=str(args.data), split="val", imgsz=args.imgsz, batch=1,
                   device=args.device, plots=False, project=str(ROOT / "runs" / "val"), name=args.name, exist_ok=True)

    info = {
        "name": args.name,
        "model_definition": args.model,
        "pretrained_weights": None,
        "parameters": n_params,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "seed": args.seed,
        "device": args.device,
        "train_time_s": round(elapsed, 1),
        "hardware": {"cpu": cpu_name(), "os": platform.platform(), "python": platform.python_version(),
                     "torch": torch.__version__, "ultralytics": ultralytics.__version__},
        "best_val": {"precision": float(val.box.mp), "recall": float(val.box.mr),
                     "mAP50": float(val.box.map50), "mAP50-95": float(val.box.map)},
    }
    (save_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
