#!/usr/bin/env python3


import argparse
import json
import os
import time
from pathlib import Path

import torch

import steer_lib as S


def sweep_pairs():
    """30 Track-1 pairs: each target with 3 fixed sources (each source used 3 times)."""
    H = S.HIGH
    return [(H[(i + k) % 10], t) for i, t in enumerate(H) for k in (1, 4, 7)]


def fmt(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--track", default="t1", choices=["t1", "t2", "t3"])
    p.add_argument("--t3-target", default="de")
    p.add_argument("--pairs", default="all", choices=["all", "sweep"])
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="layer indices; default: --layer-fracs of the model depth")
    p.add_argument("--layer-fracs", type=float, nargs="+",
                   default=[0.375, 0.5, 0.625, 0.75, 0.875])
    p.add_argument("--strengths", type=float, nargs="+", required=True)
    p.add_argument("--vectors", nargs="+", default=["source"], choices=["source", "others"])
    p.add_argument("--k", type=int, default=None, help="only first k questions (smoke test)")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("CLAS_DATA", "data")))
    p.add_argument("--vectors-root", type=Path, default=Path(os.environ.get("CLAS_VECTORS", "vectors")))
    p.add_argument("--out-root", type=Path, default=None,
                   help="default: $CLAS_GEN/<split>")
    args = p.parse_args()

    L = S.import_official()
    short = args.model.split("/")[-1]
    out_root = args.out_root or Path(os.environ.get("CLAS_GEN", "generations")) / args.split
    track_dir = args.track if args.track != "t3" else f"t3/{args.t3_target}"
    pairs = sweep_pairs() if args.pairs == "sweep" else L.pairs(args.track, args.t3_target)
    sources = sorted({s for s, _ in pairs})

    qdir = args.data_dir / f"validation_{args.split}"
    questions = {s: [l.strip() for l in (qdir / f"{s}.txt").read_text(encoding="utf-8").splitlines()
                     if l.strip()][:args.k] for s in sources}
    n_q = len(questions[sources[0]])
    print(f"{short} {args.split} {args.track}: {len(pairs)} pairs x {n_q} questions")

    means_file = args.vectors_root / short / "means.pt"
    md = torch.load(means_file, map_location="cpu")
    means = md["means"]
    if args.layers is None:
        args.layers = sorted({round(f * md["num_layers"]) for f in args.layer_fracs})
    print(f"layers {args.layers}, strengths {args.strengths}, vectors {args.vectors}")

    configs = [(v, l, c) for v in args.vectors for l in args.layers for c in args.strengths]
    todo = []
    for v, l, c in configs:
        d = out_root / short / track_dir / f"diffmean-{v}-L{l}-c{fmt(c)}"
        if all((d / f"{s}_to_{t}.json").exists() for s, t in pairs):
            print(f"skip (done): {d}")
        else:
            todo.append((v, l, c, d))
    if not todo:
        return 0

    ref_langs = S.TRACK_LANGS
    model, tok = S.load_model(args.model)
    n_layers = len(S.get_layers(model))
    texts = {s: S.chat_inputs(tok, questions[s]) for s in sources}

    for v, layer, c, d in todo:
        if not 0 <= layer < n_layers:
            print(f"!! layer {layer} out of range (model has {n_layers}); skipping")
            continue
        t0 = time.time()
        print(f"\n=== vector={v} layer={layer} strength={c} -> {d}", flush=True)
        steerer = S.Steerer(model, layer)
        items, keys = [], []
        for s, t in pairs:
            vec = c * S.steering_vector(means, layer, s, t, v, ref_langs)
            for j in range(n_q):
                items.append({"text": texts[s][j], "vec": vec})
                keys.append((s, t, j))
        try:
            outs = S.generate(model, tok, items, args.max_new_tokens, args.batch_size,
                              steerer=steerer, log_every=5)
        finally:
            steerer.remove()
        by_pair = {}
        for (s, t, j), o in zip(keys, outs):
            by_pair.setdefault((s, t), []).append((j, o))
        d.mkdir(parents=True, exist_ok=True)
        for (s, t), rows in by_pair.items():
            rows.sort()
            data = {
                "source_language": s, "target_language": t, "model": args.model,
                "method": f"diffmean:{v}:layer{layer}:strength{c}", "track": args.track,
                "steering": {"vector": v, "layer": layer, "strength": c,
                             "max_new_tokens": args.max_new_tokens, "split": args.split},
                "results": [{"question_idx": j, "input": questions[s][j], "output": o,
                             "original_prompt": questions[s][j]} for j, o in rows],
            }
            (d / f"{s}_to_{t}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
        print(f"=== done {len(items)} answers in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
