"""Evaluador comun para los tres metodos (YOLO desde cero, detector preentrenado y VLM).

Entrada: uno o varios predictions.json con el formato
  {"meta": {"method": str, "conf_threshold": float, "has_scores": bool, ...},
   "images": {id: {"width", "height", "predictions": [{"xyxy": [x1,y1,x2,y2], "conf": float}],
                   "ground_truth": [[x1,y1,x2,y2], ...]}}}

Metricas (emparejamiento voraz por confianza, cada coche real se empareja una sola vez, como en VOC):
  - En el punto de trabajo (conf >= conf_threshold, IoU >= 0.5): TP, FP, FN, precision, recall, F1
    y el IoU medio de los aciertos (calidad de ajuste de las cajas).
  - AP50 y AP50-95 (interpolacion de todos los puntos), solo si el metodo da puntuaciones.
  - Recall por tamano del coche real y P/R/F1 por tipo de escena (numero de coches, de split.csv).

Uso:
  uv run python src/evaluate.py --pred outputs/yolo_scratch/predictions.json outputs/yoloe/predictions.json \\
      --out outputs/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
SIZE_BINS = [("pequeno (<1%)", 0.0, 0.01), ("mediano (1-10%)", 0.01, 0.10), ("grande (>10%)", 0.10, 1.01)]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match_image(preds: list[dict], gt: list[list[float]], iou_thr: float) -> tuple[list[tuple[float, bool, float]], list[bool]]:
    """Devuelve [(conf, es_tp, iou)] por prediccion y [emparejado] por coche real."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["conf"])
    p = np.array([preds[i]["xyxy"] for i in order]).reshape(-1, 4)
    g = np.array(gt).reshape(-1, 4)
    ious = iou_matrix(p, g)
    gt_used = [False] * len(gt)
    out = []
    for k, i in enumerate(order):
        j = int(ious[k].argmax()) if len(gt) else -1
        best = float(ious[k, j]) if j >= 0 else 0.0
        # si el mejor coche ya esta emparejado, buscamos el siguiente disponible por encima del umbral
        if j >= 0 and gt_used[j]:
            cand = [(float(ious[k, jj]), jj) for jj in range(len(gt)) if not gt_used[jj]]
            best, j = max(cand, default=(0.0, -1))
        if j >= 0 and best >= iou_thr:
            gt_used[j] = True
            out.append((preds[i]["conf"], True, best))
        else:
            out.append((preds[i]["conf"], False, best))
    return out, gt_used


def average_precision(records: list[tuple[float, bool, float]], n_gt: int) -> tuple[float, np.ndarray, np.ndarray]:
    if not records or n_gt == 0:
        return 0.0, np.array([0.0]), np.array([0.0])
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([not r[1] for r in records])
    recall = tp / n_gt
    precision = tp / (tp + fp)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), recall, precision


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": round(p, 4), "recall": round(r, 4),
            "F1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def filter_conf(data: dict, conf: float) -> dict[str, dict]:
    return {k: {**v, "predictions": [p for p in v["predictions"] if p["conf"] >= conf]} for k, v in data["images"].items()}


def best_f1_threshold(data: dict, iou_thr: float = 0.5) -> tuple[float, float]:
    """Umbral de confianza que maximiza F1 (para elegirlo en VALIDACION)."""
    records, n_gt = [], 0
    for v in data["images"].values():
        r, _ = match_image(v["predictions"], v["ground_truth"], iou_thr)
        records += r
        n_gt += len(v["ground_truth"])
    records.sort(key=lambda r: -r[0])
    best = (0.0, 1.0)
    tp = fp = 0
    for conf, is_tp, _ in records:
        tp += is_tp
        fp += not is_tp
        f1 = 2 * tp / (tp + fp + n_gt)
        if f1 > best[0]:
            best = (f1, conf)
    return best[1], best[0]


def evaluate(data: dict, split_info: dict[str, dict]) -> dict:
    meta = data["meta"]
    has_scores = meta.get("has_scores", True)
    images = data["images"]
    n_gt = sum(len(v["ground_truth"]) for v in images.values())

    ap = {}
    if has_scores:
        for t in IOU_THRESHOLDS:
            recs = [r for v in images.values() for r in match_image(v["predictions"], v["ground_truth"], t)[0]]
            ap[round(float(t), 2)] = average_precision(recs, n_gt)
    op = filter_conf(data, meta.get("conf_threshold", 0.0))

    tp = fp = fn = 0
    matched_ious, per_image, size_hits = [], {}, {name: [0, 0] for name, _, _ in SIZE_BINS}
    scene = {}
    for img_id, v in op.items():
        recs, used = match_image(v["predictions"], v["ground_truth"], 0.5)
        i_tp = sum(r[1] for r in recs)
        i_fp = len(recs) - i_tp
        i_fn = used.count(False)
        tp, fp, fn = tp + i_tp, fp + i_fp, fn + i_fn
        matched_ious += [r[2] for r in recs if r[1]]
        per_image[img_id] = prf(i_tp, i_fp, i_fn)
        for box, hit in zip(v["ground_truth"], used):
            rel = (box[2] - box[0]) * (box[3] - box[1]) / (v["width"] * v["height"])
            name = next(n for n, lo, hi in SIZE_BINS if lo <= rel < hi)
            size_hits[name][0] += hit
            size_hits[name][1] += 1
        key = split_info.get(img_id, {}).get("count_bin", "?")
        s = scene.setdefault(key, [0, 0, 0])
        s[0], s[1], s[2] = s[0] + i_tp, s[1] + i_fp, s[2] + i_fn

    result = {
        "method": meta.get("method", "?"),
        "n_images": len(images), "n_cars": n_gt,
        "conf_threshold": meta.get("conf_threshold"),
        "operating_point": prf(tp, fp, fn),
        "mean_iou_tp": round(float(np.mean(matched_ious)), 4) if matched_ious else None,
        "AP50": round(ap[0.5][0], 4) if ap else None,
        "AP50-95": round(float(np.mean([a[0] for a in ap.values()])), 4) if ap else None,
        "recall_by_size": {k: {"hits": h, "total": t, "recall": round(h / t, 4) if t else None} for k, (h, t) in size_hits.items()},
        "by_scene_n_cars": {k: prf(*v) for k, v in sorted(scene.items())},
        "per_image": per_image,
    }
    if ap:
        result["_pr_curve"] = (ap[0.5][1].tolist(), ap[0.5][2].tolist())
    return result


def load_split_info(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        return {row["image_id"]: row for row in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "comparison")
    ap.add_argument("--split-csv", type=Path, default=ROOT / "dataset" / "split.csv")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    split_info = load_split_info(args.split_csv)

    results = []
    for p in args.pred:
        data = json.loads(p.read_text())
        res = evaluate(data, split_info)
        timing = p.parent / "timing.json"
        if timing.exists():
            res["mean_ms_per_image"] = json.loads(timing.read_text()).get("mean_ms_per_image")
        results.append(res)

    fig, ax = plt.subplots(figsize=(6, 5))
    rows = []
    for i, r in enumerate(results):
        if "_pr_curve" in r:
            rec, prec = r.pop("_pr_curve")
            ax.plot(rec, prec, color=f"C{i}", label=f"{r['method']} (AP50={r['AP50']:.3f})")
        op = r["operating_point"]
        ax.scatter([op["recall"]], [op["precision"]], color=f"C{i}", marker="o", s=60, edgecolor="black", zorder=3,
                   label=None if "AP50" in r and r["AP50"] is not None else f"{r['method']} (sin puntuaciones)")
        rows.append({"method": r["method"], "conf_threshold": r["conf_threshold"], **op, "mean_iou_tp": r["mean_iou_tp"],
                     "AP50": r["AP50"], "AP50-95": r["AP50-95"], "mean_ms_per_image": r.get("mean_ms_per_image"),
                     **{f"recall_{k.split()[0]}": v["recall"] for k, v in r["recall_by_size"].items()}})
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.set_title("Curva precision-recall (IoU 0.5); punto = umbral elegido")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "pr_curves.png", dpi=130)
    plt.close(fig)

    (args.out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
