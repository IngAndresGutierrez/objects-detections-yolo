"""Compara experimentos de entrenamiento y elige el modelo final usando SOLO validacion.

- Superpone las curvas (loss de train/val y mAP de val) de todos los experimentos.
- Revalida el best.pt de cada experimento con batch=1 (cada imagen con su propio letterbox):
  con lotes mayores el "rect batching" rellena las imagenes segun con cuales se agrupen y,
  con solo 10 imagenes de validacion, la metrica cambia (p. ej. 0.145 vs 0.206 de mAP50).
- Criterio: mayor mAP50 en validacion (metrica estandar de VOC); desempate por mAP50-95.
- Fija el umbral de confianza del modelo elegido como el que maximiza F1 en validacion.
- Copia los pesos elegidos a models/yolo26n_scratch_best.pt y escribe models/selection.json.

Uso:
  uv run python src/select_model.py --runs img416 img640
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
PANELS = [
    ("train/box_loss", "val/box_loss", "Box loss"),
    ("train/cls_loss", "val/cls_loss", "Class loss"),
    (None, "metrics/mAP50(B)", "mAP50 (val)"),
    (None, "metrics/mAP50-95(B)", "mAP50-95 (val)"),
]


def plot_curves(runs: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(18, 4))
    for ax, (train_col, val_col, title) in zip(axes, PANELS):
        for i, (name, df) in enumerate(runs.items()):
            color = f"C{i}"
            if train_col:
                ax.plot(df["epoch"], df[train_col], color=color, linestyle="--", alpha=0.7, label=f"{name} train")
            ax.plot(df["epoch"], df[val_col], color=color, label=f"{name} val" if train_col else name)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "dataset" / "data.yaml")
    args = ap.parse_args()

    train_dir = ROOT / "runs" / "train"
    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)

    curves, infos = {}, {}
    for name in args.runs:
        df = pd.read_csv(train_dir / name / "results.csv")
        df.columns = df.columns.str.strip()
        curves[name] = df
        infos[name] = json.loads((train_dir / name / "run_info.json").read_text())

    plot_curves(curves, out_dir / "training_curves_comparison.png")

    rows = []
    for name, info in infos.items():
        df = curves[name]
        best_epoch = int(df.loc[df["metrics/mAP50(B)"].idxmax(), "epoch"])
        v = YOLO(train_dir / name / "weights" / "best.pt").val(
            data=str(args.data), split="val", imgsz=info["imgsz"], batch=1, device=info["device"],
            plots=False, verbose=False, project=str(ROOT / "runs" / "val"), name=f"{name}_b1", exist_ok=True)
        rows.append({"run": name, "imgsz": info["imgsz"], "epochs": info["epochs"], "batch": info["batch"],
                     "train_time_s": info["train_time_s"], "best_epoch_curve": best_epoch,
                     "val_precision": round(float(v.box.mp), 4), "val_recall": round(float(v.box.mr), 4),
                     "val_mAP50": round(float(v.box.map50), 4), "val_mAP50-95": round(float(v.box.map), 4)})
    table = pd.DataFrame(rows).sort_values(["val_mAP50", "val_mAP50-95"], ascending=False)
    table.to_csv(out_dir / "configs_comparison.csv", index=False)
    print(table.to_string(index=False))

    chosen = table.iloc[0]
    info = infos[chosen["run"]]
    weights = train_dir / chosen["run"] / "weights" / "best.pt"

    val = YOLO(weights).val(data=str(args.data), split="val", imgsz=info["imgsz"], batch=1,
                            conf=0.001, device=info["device"], plots=False,
                            project=str(ROOT / "runs" / "val"), name=f"{chosen['run']}_thr", exist_ok=True)
    f1 = np.asarray(val.box.f1_curve).reshape(-1)
    px = np.linspace(0, 1, len(f1))
    conf = float(px[int(f1.argmax())])

    shutil.copy(weights, out_dir / "yolo26n_scratch_best.pt")
    selection = {"chosen_run": chosen["run"], "criterion": "max val mAP50, tie-break val mAP50-95",
                 "imgsz": info["imgsz"], "conf_threshold": round(conf, 3), "val_best_f1": round(float(f1.max()), 4),
                 "weights": "models/yolo26n_scratch_best.pt", "comparison": rows}
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    print(f"\nElegido: {chosen['run']} | umbral conf (max F1 val) = {conf:.3f} | F1 = {f1.max():.3f}")


if __name__ == "__main__":
    main()
