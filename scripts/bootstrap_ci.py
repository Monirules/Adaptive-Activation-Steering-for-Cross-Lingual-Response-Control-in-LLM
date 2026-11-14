#!/usr/bin/env python3

import argparse
import csv

import numpy as np

import result_sets as R

COMPARISONS = [
    ("t1", "diffmean_l20", "prompting"),
    ("t1", "diffmean_l16", "diffmean_l20"),
    ("t1", "tuned", "diffmean_l20"),
    ("t1", "tuned", "diffmean_l16"),
    ("t1", "adaptive", "diffmean_l16"),
    ("t1", "adaptive", "tuned"),
    ("t2", "diffmean_l20", "prompting"),
    ("t2", "diffmean_l16", "prompting"),
    ("t2", "tuned", "diffmean_l20"),
    ("t2", "tuned", "diffmean_l16"),
    ("t2", "adaptive", "diffmean_l16"),
    ("t2", "tuned_adaptive", "tuned"),
    ("t2", "tuned_adaptive", "prompting"),
    ("t3", "diffmean_l20", "prompting"),
    ("t3", "tuned", "diffmean_l20"),
    ("t3", "adaptive", "diffmean_l16"),
]


def pair_arrays(scores: dict, keys: list):
    bp = scores["by_pair"]
    return (np.array([bp[k]["lfs"] for k in keys]), np.array([bp[k]["or"] for k in keys]))


def lss(lfs: np.ndarray, orr: np.ndarray) -> np.ndarray:
    s = lfs + orr
    return np.where(s > 0, 2 * lfs * orr / np.where(s > 0, s, 1), 0.0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=R.LLAMA)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2027)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for track, a, b in COMPARISONS:
        sa = R.load_scores(args.model, track, R.METHODS[a])
        sb = R.load_scores(args.model, track, R.METHODS[b])
        if sa is None or sb is None:
            print(f"skip {track}: {a} vs {b} (missing scores)")
            continue
        keys = sorted(set(sa["by_pair"]) & set(sb["by_pair"]))
        fa, oa = pair_arrays(sa, keys)
        fb, ob = pair_arrays(sb, keys)
        n = len(keys)
        idx = rng.integers(0, n, size=(args.n_boot, n))
        la = lss(fa[idx].mean(1), oa[idx].mean(1))
        lb = lss(fb[idx].mean(1), ob[idx].mean(1))
        d = la - lb
        obs = R.harmonic(fa.mean(), oa.mean()) - R.harmonic(fb.mean(), ob.mean())
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_two = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        rows.append({
            "track": R.TRACK_LABELS[track], "method": R.METHOD_LABELS[a],
            "baseline": R.METHOD_LABELS[b], "pairs": n,
            "LSS_method": round(R.harmonic(fa.mean(), oa.mean()), 3),
            "LSS_baseline": round(R.harmonic(fb.mean(), ob.mean()), 3),
            "delta_LSS": round(obs, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
            "delta_LFS": round(fa.mean() - fb.mean(), 3), "delta_OR": round(oa.mean() - ob.mean(), 3),
            "p_value": round(p_two, 4), "significant": "yes" if lo > 0 or hi < 0 else "no",
        })

    if not rows:
        raise SystemExit("no comparisons could be computed")
    out_csv = R.RESULTS / "significance.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = [f"# Paired bootstrap over language pairs ({args.model}, {args.n_boot} resamples, 95% CI)", "",
          "| track | method | baseline | LSS | baseline LSS | delta LSS | 95% CI | delta LFS | delta OR | p | significant |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['track']} | {r['method']} | {r['baseline']} | {r['LSS_method']:.3f} | "
                  f"{r['LSS_baseline']:.3f} | {r['delta_LSS']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                  f"{r['delta_LFS']:+.3f} | {r['delta_OR']:+.3f} | {r['p_value']:.4f} | {r['significant']} |")
    (R.RESULTS / "significance.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwritten -> {out_csv}, {R.RESULTS / 'significance.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
