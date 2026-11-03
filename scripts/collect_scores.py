#!/usr/bin/env python3
"""Collect every scores.json written by the official evaluate.py into tables.

Looks under $CLAS_GEN/<split>/<model>/<track...>/<method>/scores.json and writes
  results/summary.csv          one row per (split, model, track, method)
  results/by_target.csv        one row per target language
  results/summary.md           tables for the report
  results/raw/...              a copy of each scores.json

Usage
  python scripts/collect_scores.py                 # dev and test
  python scripts/collect_scores.py --include-smoke
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gen-root", type=Path,
                   default=Path(os.environ.get("CLAS_GEN", "generations")))
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--include-smoke", action="store_true")
    args = p.parse_args()

    rows, targets = [], []
    for f in sorted(args.gen_root.rglob("scores.json")):
        parts = f.relative_to(args.gen_root).parts[:-1]
        if len(parts) < 4:
            continue
        split, model, method = parts[0], parts[1], parts[-1]
        track = "/".join(parts[2:-1])
        if split.startswith("smoke") and not args.include_smoke:
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        o = s["overall"]
        unparsed = sum(v.get("n_unparsed_judge", 0) for v in s["by_pair"].values())
        items = sum(v.get("n_items", 0) for v in s["by_pair"].values())
        rows.append({"split": split, "model": model, "track": track, "method": method,
                     "LFS": round(o["lfs"], 4), "OR": round(o["or"], 4),
                     "LSS": round(o["lss"], 4), "pairs": o["n_pairs"], "answers": items,
                     "judge_unparsed": unparsed})
        for tgt, t in s["by_target_language"].items():
            targets.append({"split": split, "model": model, "track": track, "method": method,
                            "target": tgt, "LFS": round(t["lfs"], 4), "OR": round(t["or"], 4),
                            "LSS": round(t["lss"], 4)})
        dst = args.out / "raw" / Path(*parts) / "scores.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)

    if not rows:
        print(f"no scores.json found under {args.gen_root}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary.csv", rows), ("by_target.csv", targets)):
        with open(args.out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    md = ["# Results", ""]
    for split in sorted({r["split"] for r in rows}):
        md += [f"## {split}", "",
               "| model | track | method | LFS | OR | LSS | pairs | answers | judge unparsed |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted((r for r in rows if r["split"] == split),
                        key=lambda r: (r["model"], r["track"], r["method"])):
            md.append(f"| {r['model']} | {r['track']} | {r['method']} | {r['LFS']:.3f} | "
                      f"{r['OR']:.3f} | **{r['LSS']:.3f}** | {r['pairs']} | {r['answers']} | "
                      f"{r['judge_unparsed']} |")
        md.append("")
    (args.out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"written -> {args.out}/summary.csv, by_target.csv, summary.md, raw/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
