#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"diffmean-(?P<v>\w+)-L(?P<l>\d+)-c(?P<c>[\dp.]+)$")


def load_rows(model_dir: Path) -> dict:
    rows = {}
    for f in sorted(model_dir.glob("t[12]/diffmean-*/scores.json")):
        m = PAT.match(f.parent.name)
        if not m:
            continue
        key = (m["v"], int(m["l"]), float(m["c"].replace("p", ".")))
        by_target = json.loads(f.read_text(encoding="utf-8"))["by_target_language"]
        for tgt, sc in by_target.items():
            rows.setdefault(tgt, {})[key] = sc
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="short name, e.g. Llama-3.1-8B-Instruct")
    p.add_argument("--sweep-root", type=Path,
                   default=Path(os.environ.get("CLAS_GEN", "generations")) / "sweep")
    p.add_argument("--best-file", type=Path, default=REPO_ROOT / "results" / "best_config.json")
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    best = json.loads(args.best_file.read_text(encoding="utf-8"))[args.model]
    gkey = (best["vector"], int(best["layer"]), float(best["strength"]))
    rows = load_rows(args.sweep_root / args.model)
    if not rows:
        raise SystemExit(f"no sweep scores under {args.sweep_root / args.model}")

    targets = {}
    print(f"{args.model}: global setting vector={gkey[0]} layer={gkey[1]} strength={gkey[2]:g}")
    print(f"{'target':6} {'choice':7} {'vector':7} {'layer':>5} {'c':>5} {'dev LSS':>8} {'global':>8} {'tried':>5}")
    for tgt in sorted(rows):
        cand = rows[tgt]
        bkey = max(cand, key=lambda k: cand[k]["lss"])
        g = cand.get(gkey)
        if g is not None and cand[bkey]["lss"] - g["lss"] < args.margin:
            key, choice = gkey, "global"
        else:
            key, choice = bkey, "tuned"
        sc = cand[key]
        targets[tgt] = {"vector": key[0], "layer": key[1], "strength": key[2], "choice": choice,
                        "dev_lfs": round(sc["lfs"], 4), "dev_or": round(sc["or"], 4),
                        "dev_lss": round(sc["lss"], 4),
                        "global_dev_lss": None if g is None else round(g["lss"], 4),
                        "settings_tried": len(cand)}
        gl = "   -" if g is None else f"{g['lss']:8.3f}"
        print(f"{tgt:6} {choice:7} {key[0]:7} {key[1]:5d} {key[2]:5g} {sc['lss']:8.3f} {gl:>8} {len(cand):5d}")

    out = args.out or REPO_ROOT / "results" / f"tuned_config_{args.model}.json"
    out.write_text(json.dumps({"model": args.model, "margin": args.margin,
                               "global": {"vector": gkey[0], "layer": gkey[1], "strength": gkey[2]},
                               "targets": targets}, indent=2), encoding="utf-8")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
