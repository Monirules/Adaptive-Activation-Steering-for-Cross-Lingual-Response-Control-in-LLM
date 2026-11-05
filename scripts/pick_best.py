#!/usr/bin/env python3


import argparse
import csv
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"diffmean-(?P<v>\w+)-L(?P<l>\d+)-c(?P<c>[\dp.]+)$")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path,
                   default=Path(os.environ.get("CLAS_GEN", "generations")) / "sweep")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    best_all = {}
    best_file = args.out / "best_config.json"
    if best_file.exists():
        best_all = json.loads(best_file.read_text(encoding="utf-8"))

    for mdir in sorted(d for d in args.root.iterdir() if d.is_dir()):
        rows = []
        for f in sorted(mdir.glob("t1/diffmean-*/scores.json")):
            m = PAT.match(f.parent.name)
            if not m:
                continue
            o = json.loads(f.read_text(encoding="utf-8"))["overall"]
            rows.append({"vector": m["v"], "layer": int(m["l"]),
                         "strength": float(m["c"].replace("p", ".")),
                         "LFS": round(o["lfs"], 4), "OR": round(o["or"], 4),
                         "LSS": round(o["lss"], 4), "pairs": o["n_pairs"]})
        if not rows:
            continue
        rows.sort(key=lambda r: (r["vector"], r["layer"], r["strength"]))
        with open(args.out / f"sweep_{mdir.name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        best = max(rows, key=lambda r: r["LSS"])
        best_all[mdir.name] = best
        print(f"\n{mdir.name}: {len(rows)} configs")
        print(f"{'vector':8} {'layer':>5} {'c':>5} {'LFS':>6} {'OR':>6} {'LSS':>6}")
        for r in rows:
            mark = "  <-- best" if r is best else ""
            print(f"{r['vector']:8} {r['layer']:5d} {r['strength']:5g} {r['LFS']:6.3f} "
                  f"{r['OR']:6.3f} {r['LSS']:6.3f}{mark}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            vectors = sorted({r["vector"] for r in rows})
            strengths = sorted({r["strength"] for r in rows})
            fig, axes = plt.subplots(len(vectors), 3, figsize=(13, 3.6 * len(vectors)), squeeze=False)
            for i, v in enumerate(vectors):
                for j, metric in enumerate(("LSS", "LFS", "OR")):
                    ax = axes[i][j]
                    for c in strengths:
                        pts = sorted((r["layer"], r[metric]) for r in rows
                                     if r["vector"] == v and r["strength"] == c)
                        if pts:
                            ax.plot(*zip(*pts), marker="o", label=f"c={c:g}")
                    ax.set_title(f"{metric}  (vector: target minus {v})")
                    ax.set_xlabel("layer")
                    ax.set_ylim(-0.02, 1.02)
                    ax.grid(alpha=0.3)
                    if j == 0:
                        ax.legend(fontsize=8)
            fig.suptitle(f"{mdir.name}: DiffMean dev sweep (20 questions, 30 pairs)")
            fig.tight_layout()
            fig.savefig(args.out / f"sweep_{mdir.name}.png", dpi=150)
            plt.close(fig)
        except ImportError:
            print("(matplotlib not installed, no figure)")

    best_file.write_text(json.dumps(best_all, indent=2), encoding="utf-8")
    print(f"\nbest configs -> {best_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
