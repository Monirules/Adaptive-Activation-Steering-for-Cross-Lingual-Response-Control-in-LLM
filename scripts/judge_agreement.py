#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import steer_lib as S

REPO_ROOT = Path(__file__).resolve().parents[1]


def kappa(a: list, b: list, weighted: bool) -> float:
    cats = [0, 1, 2]
    n = len(a)
    if n == 0:
        return float("nan")
    obs = [[0] * 3 for _ in cats]
    for x, y in zip(a, b):
        obs[x][y] += 1
    pa = [sum(obs[i]) / n for i in cats]
    pb = [sum(obs[i][j] for i in cats) / n for j in cats]
    w = [[((i - j) ** 2) / 4 if weighted else (0 if i == j else 1) for j in cats] for i in cats]
    do = sum(w[i][j] * obs[i][j] / n for i in cats for j in cats)
    de = sum(w[i][j] * pa[i] * pb[j] for i in cats for j in cats)
    return 1.0 if de == 0 else 1 - do / de


def describe(name: str, a: list, b: list) -> dict:
    n = len(a)
    return {"pair": name, "n": n,
            "exact_agreement": round(sum(x == y for x, y in zip(a, b)) / n, 3) if n else None,
            "cohen_kappa": round(kappa(a, b, False), 3),
            "quadratic_kappa": round(kappa(a, b, True), 3),
            "mean_first": round(sum(a) / (2 * n), 3) if n else None,
            "mean_second": round(sum(b) / (2 * n), 3) if n else None}


def rating(v: str):
    v = (v or "").strip()
    return int(v) if v in ("0", "1", "2") else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="hi")
    p.add_argument("--dir", type=Path, default=REPO_ROOT / "results" / "annotation")
    p.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-model-len", type=int, default=8192)
    args = p.parse_args()

    key = json.loads((args.dir / f"{args.target}_key.json").read_text(encoding="utf-8"))
    with open(args.dir / f"{args.target}_ratings.csv", encoding="utf-8-sig", newline="") as fh:
        sheet = {int(r["item_id"]): r for r in csv.DictReader(fh)}

    gen_dir = Path(key["gen_dir"])
    files = sorted({gen_dir / it["file"] for it in key["items"]})

    S.import_official()
    spec = importlib.util.spec_from_file_location("ev", S.official_dir() / "evaluate.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    import torch
    from vllm import LLM, SamplingParams
    judge = LLM(model=args.judge_model, tensor_parallel_size=max(1, torch.cuda.device_count()),
                max_model_len=args.max_model_len, dtype="bfloat16", gpu_memory_utilization=0.9,
                trust_remote_code=True)
    judged = E.judge_all(files, judge, SamplingParams(temperature=0.0, max_tokens=512, seed=0))

    rows = []
    for it in key["items"]:
        r = sheet.get(it["item_id"], {})
        rows.append({"item_id": it["item_id"], "judge": judged[gen_dir / it["file"]][it["index"]],
                     "rater_1": rating(r.get("rater_1")), "rater_2": rating(r.get("rater_2"))})

    def both(x, y):
        keep = [(r[x], r[y]) for r in rows if r[x] is not None and r[y] is not None]
        return [k[0] for k in keep], [k[1] for k in keep]

    results = [describe("rater_1 vs rater_2", *both("rater_1", "rater_2")),
               describe("rater_1 vs judge", *both("rater_1", "judge")),
               describe("rater_2 vs judge", *both("rater_2", "judge"))]
    avg = []
    for r in rows:
        hs = [r[k] for k in ("rater_1", "rater_2") if r[k] is not None]
        if hs and r["judge"] is not None:
            avg.append((round(sum(hs) / len(hs)), r["judge"]))
    results.append(describe("human average (rounded) vs judge", [a for a, _ in avg], [b for _, b in avg]))

    with open(args.dir / f"{args.target}_judge_vs_human.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "rater_1", "rater_2", "judge"])
        w.writeheader()
        w.writerows(rows)
    md = ["| comparison | items | exact agreement | Cohen kappa | quadratic kappa | OR first | OR second |",
          "|---|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['pair']} | {r['n']} | {r['exact_agreement']} | {r['cohen_kappa']} | "
                  f"{r['quadratic_kappa']} | {r['mean_first']} | {r['mean_second']} |")
    (args.dir / f"{args.target}_agreement.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
