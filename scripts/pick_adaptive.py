#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"adaptive-(?P<mode>gate|decay)-w(?P<w>\d+)-t(?P<t>[\dp]+)-l(?P<late>[\dp]+)"
                 r"-(?P<v>\w+)-L(?P<l>\d+)-c(?P<c>[\dp]+)$")


def num(x: str) -> float:
    return float(x.replace("p", "."))


def fmt(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def overall(f: Path) -> dict:
    return json.loads(f.read_text(encoding="utf-8"))["overall"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="short name, e.g. Llama-3.1-8B-Instruct")
    p.add_argument("--sweep-root", type=Path,
                   default=Path(os.environ.get("CLAS_GEN", "generations")) / "sweep")
    p.add_argument("--best-file", type=Path, default=REPO_ROOT / "results" / "best_config.json")
    p.add_argument("--tracks", nargs="+", default=["t1", "t2"])
    p.add_argument("--ema", type=float, default=0.8)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    root = args.sweep_root / args.model
    best = json.loads(args.best_file.read_text(encoding="utf-8"))[args.model]
    fixed_name = f"diffmean-{best['vector']}-L{best['layer']}-c{fmt(float(best['strength']))}"

    settings = {}
    for tr in args.tracks:
        for f in sorted(root.glob(f"{tr}/adaptive-*/scores.json")):
            m = PAT.match(f.parent.name)
            if not m:
                continue
            key = (m["mode"], int(m["w"]), num(m["t"]), num(m["late"]), m["v"], int(m["l"]), num(m["c"]))
            settings.setdefault(key, {})[tr] = overall(f)

    fixed = {}
    for tr in args.tracks:
        f = root / tr / fixed_name / "scores.json"
        if f.exists():
            fixed[tr] = overall(f)

    complete = {k: v for k, v in settings.items() if all(tr in v for tr in args.tracks)}
    if not complete:
        raise SystemExit(f"no adaptive setting has scores for all of {args.tracks} under {root}")

    def mean(d, metric):
        return sum(d[tr][metric] for tr in args.tracks) / len(args.tracks)

    head = " ".join(f"{tr + ' LSS':>8}" for tr in args.tracks)
    print(f"{args.model}: dev results, mean over {', '.join(args.tracks)}")
    print(f"{'setting':34} {head} {'mean LFS':>9} {'mean OR':>8} {'mean LSS':>9}")
    if len(fixed) == len(args.tracks):
        cols = " ".join(f"{fixed[tr]['lss']:8.3f}" for tr in args.tracks)
        print(f"{'fixed ' + fixed_name:34} {cols} {mean(fixed, 'lfs'):9.3f} {mean(fixed, 'or'):8.3f} "
              f"{mean(fixed, 'lss'):9.3f}")
    ranked = sorted(complete, key=lambda k: mean(complete[k], "lss"), reverse=True)
    for k in ranked:
        name = f"{k[0]} w={k[1]} tau={k[2]:g} late={k[3]:g}"
        cols = " ".join(f"{complete[k][tr]['lss']:8.3f}" for tr in args.tracks)
        print(f"{name:34} {cols} {mean(complete[k], 'lfs'):9.3f} {mean(complete[k], 'or'):8.3f} "
              f"{mean(complete[k], 'lss'):9.3f}")

    k = ranked[0]
    fixed_lss = mean(fixed, "lss") if len(fixed) == len(args.tracks) else None
    cfg = {"model": args.model, "mode": k[0], "warmup": k[1], "tau": k[2], "late": k[3],
           "ema": args.ema, "vector": k[4], "layer": k[5], "strength": k[6],
           "dev_tracks": args.tracks,
           "dev_lss": round(mean(complete[k], "lss"), 4),
           "dev_lfs": round(mean(complete[k], "lfs"), 4),
           "dev_or": round(mean(complete[k], "or"), 4),
           "fixed_dev_lss": None if fixed_lss is None else round(fixed_lss, 4)}
    out = args.out or REPO_ROOT / "results" / f"adaptive_config_{args.model}.json"
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\nbest: {k[0]} warmup={k[1]} tau={k[2]:g} late={k[3]:g} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
