#!/usr/bin/env python3


import argparse
import importlib.util
from pathlib import Path

import steer_lib as S


def load_official_evaluate():
    path = S.official_dir() / "evaluate.py"
    spec = importlib.util.spec_from_file_location("clas_official_evaluate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--redo", action="store_true")
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = p.parse_args()

    S.import_official()
    E = load_official_evaluate()
    from clas.io_utils import read_json

    dirs = sorted({f.parent for f in args.root.rglob("*_to_*.json")})
    dirs = [d for d in dirs if args.redo or not (d / "scores.json").exists()]
    if not dirs:
        print("nothing to score")
        return 0
    files = {d: sorted(d.glob("*_to_*.json")) for d in dirs}
    all_files = [f for d in dirs for f in files[d]]
    print(f"scoring {len(dirs)} folders, {len(all_files)} pair files")

    lid, mapping = E.load_lid(), E.L.flores_to_code()
    lfs = {}
    for f in all_files:
        data = read_json(f)
        hits = [E.predict_lang(lid, r["output"], mapping, 0.9) == data["target_language"]
                for r in data["results"]]
        lfs[f] = sum(hits) / len(hits)

    if args.skip_judge:  # print LFS only; write nothing, so the full scoring still runs later
        for d in dirs:
            v = [lfs[f] for f in files[d]]
            print(f"LFS {sum(v) / len(v):.3f}  {d}")
        return 0
    else:
        import torch
        from vllm import LLM, SamplingParams
        judge = LLM(model=args.judge_model, tensor_parallel_size=max(1, torch.cuda.device_count()),
                    max_model_len=args.max_model_len, dtype="bfloat16",
                    gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
        judged = E.judge_all(all_files, judge, SamplingParams(temperature=0.0, max_tokens=512, seed=0))

    for d in dirs:
        pair_scores = {}
        for f in files[d]:
            data = read_json(f)
            valid = [s for s in judged[f] if s is not None]
            orl = (sum(valid) / (2 * len(valid))) if valid else 0.0
            pair_scores[f"{data['source_language']}_to_{data['target_language']}"] = {
                "source": data["source_language"], "target": data["target_language"],
                "lfs": lfs[f], "or": orl, "lss": E.harmonic(lfs[f], orl),
                "n_items": len(data["results"]), "n_unparsed_judge": len(judged[f]) - len(valid),
            }
        print(f"\n### {d}")
        E.report(pair_scores, d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
