"""(NO ENTREGABLE) Convierte las respuestas del VLM (pegadas desde la web) en predicciones evaluables.

Respuestas en outputs/vlm/responses/<split>/<prompt>/<id>.txt; prompts en outputs/vlm/PROMPTS.md.
Formatos de caja reconocidos:
  - "box_2d": [ymin, xmin, ymax, xmax] normalizado 0-1000 (Gemini)
  - "bbox_2d": [x1, y1, x2, y2] normalizado 0-1000 (Qwen3-VL)  -> --bbox-mode norm1000 (defecto)
                                    o en pixeles (Qwen2.5-VL)   -> --bbox-mode pixels
El VLM no da puntuaciones: cada caja tiene conf=1.0 y solo se calculan P/R/F1 e IoU (has_scores=False).

Uso:
  uv run python part2_no_entregable/vlm_parse.py val            # evalua P1/P2 y elige prompt por F1
  uv run python part2_no_entregable/vlm_parse.py test           # prompt elegido + P3 sobre test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import evaluate, load_split_info  # noqa: E402
from predict import read_gt, side_by_side  # noqa: E402

VLM = ROOT / "outputs" / "vlm"


def load_prompts() -> dict[str, str]:
    text = (VLM / "PROMPTS.md").read_text()
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"### (\w+)[^\n]*\n+```\n(.*?)```", text, re.S)}


def extract_json(raw: str) -> list[dict]:
    raw = re.sub(r"```(?:json)?", "", raw)
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < start:
        raise ValueError("no se encuentra una lista JSON en la respuesta")
    items = json.loads(raw[start:end + 1])
    return [it for it in items if isinstance(it, dict)]


def to_xyxy(item: dict, w: int, h: int, bbox_mode: str) -> list[float] | None:
    if "box_2d" in item:
        y1, x1, y2, x2 = item["box_2d"]
        box = [x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h]
    elif "bbox_2d" in item:
        x1, y1, x2, y2 = item["bbox_2d"]
        box = [x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h] if bbox_mode == "norm1000" else [x1, y1, x2, y2]
    else:
        return None
    x1, y1, x2, y2 = box
    box = [max(0.0, min(x1, x2)), max(0.0, min(y1, y2)), min(float(w), max(x1, x2)), min(float(h), max(y1, y2))]
    return [round(v, 1) for v in box] if box[2] > box[0] and box[3] > box[1] else None


def parse_prompt(split: str, prompt: str, prompt_text: str, bbox_mode: str) -> tuple[dict, list[dict]]:
    folder = VLM / "responses" / split / prompt
    images, log, missing = {}, [], []
    for txt in sorted(folder.glob("*.txt")):
        raw = txt.read_text().strip()
        if not raw:
            missing.append(txt.stem)
            continue
        img = Image.open(ROOT / "dataset" / "images" / split / f"{txt.stem}.jpg")
        entry: dict = {"image": txt.stem, "prompt": prompt_text, "raw_response": raw}
        try:
            items = extract_json(raw)
            preds = []
            for it in items:
                box = to_xyxy(it, img.width, img.height, bbox_mode)
                if box:
                    extra = {k: v for k, v in it.items() if k not in ("box_2d", "bbox_2d")}
                    preds.append({"xyxy": box, "conf": 1.0, **extra})
            entry["parsed_boxes"] = len(preds)
        except (ValueError, json.JSONDecodeError) as e:
            preds, entry["parse_error"] = [], str(e)  # respuesta no parseable = sin detecciones
        gt = read_gt(ROOT / "dataset" / "labels" / split / f"{txt.stem}.txt", img.width, img.height)
        images[txt.stem] = {"width": img.width, "height": img.height, "predictions": preds,
                            "ground_truth": [[round(v, 1) for v in b] for b in gt]}
        log.append(entry)
    if missing:
        raise SystemExit(f"Faltan respuestas en {folder.relative_to(ROOT)}: {missing}")
    meta = {"method": f"VLM [{prompt}]", "conf_threshold": 0.0, "has_scores": False, "split": split,
            "bbox_mode": bbox_mode}
    return {"meta": meta, "images": images}, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["val", "test"])
    ap.add_argument("--bbox-mode", choices=["norm1000", "pixels"], default="norm1000")
    args = ap.parse_args()
    prompts, split_info = load_prompts(), load_split_info(ROOT / "dataset" / "split.csv")

    if args.split == "val":
        rows = []
        for p in ("P1_simple", "P2_voc"):
            data, log = parse_prompt("val", p, prompts[p], args.bbox_mode)
            res = evaluate(data, split_info)
            out = VLM / f"val_{p}"
            out.mkdir(exist_ok=True)
            (out / "predictions.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
            (out / "prompts_and_responses.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
            rows.append({"prompt": p, "mean_iou_tp": res["mean_iou_tp"], **res["operating_point"],
                         "parse_errors": sum("parse_error" in e for e in log)})
            print(json.dumps(rows[-1]))
        best = max(rows, key=lambda r: (r["F1"], r["mean_iou_tp"] or 0))
        (VLM / "prompt_selection_val.json").write_text(json.dumps(
            {"criterion": "max val F1 (IoU 0.5), desempate IoU medio", "chosen": best, "all": rows}, indent=2))
        (VLM / "responses" / "test" / best["prompt"]).mkdir(parents=True, exist_ok=True)
        for img in (ROOT / "dataset" / "images" / "test").glob("*.jpg"):
            (VLM / "responses" / "test" / best["prompt"] / f"{img.stem}.txt").touch()
        print(f"\nElegido en validacion: {best['prompt']} -> rellena responses/test/{best['prompt']}/ y P3_descriptivo/")
        return

    chosen = json.loads((VLM / "prompt_selection_val.json").read_text())["chosen"]["prompt"]
    for p, out in ((chosen, VLM), ("P3_descriptivo", VLM / "test_P3_descriptivo")):
        data, log = parse_prompt("test", p, prompts[p], args.bbox_mode)
        out.mkdir(exist_ok=True)
        (out / "predictions.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
        (out / "prompts_and_responses.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
        (out / "gallery").mkdir(exist_ok=True)
        for img_id, v in data["images"].items():
            img = Image.open(ROOT / "dataset" / "images" / "test" / f"{img_id}.jpg").convert("RGB")
            side_by_side(img, v["predictions"], v["ground_truth"], show_conf=False).save(out / "gallery" / f"{img_id}.jpg", quality=90)
        print(p, json.dumps(evaluate(data, split_info)["operating_point"]))


if __name__ == "__main__":
    main()
