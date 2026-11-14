#!/usr/bin/env python3

import argparse
import csv
import json
import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBRIC = """Hindi output check: rating instructions

Each row is one answer written by the model in Hindi. The question is shown in its original (source) language.
Rate only the content of the answer, the same way the official judge does.

0 = unrelated to the question, gibberish, or broken repeated text
1 = somewhat related, but incomplete, partly off-topic, or of little use
2 = clearly related to the question, coherent, and useful

Write 0, 1 or 2 in your column (rater_1 or rater_2). Rate on your own, without looking at the other rater's column.
Use the notes column for anything unusual, for example mixed languages or text that loops.
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gen-dir", type=Path, default=Path(os.environ.get("CLAS_GEN", "generations")) /
                   "test" / "Llama-3.1-8B-Instruct" / "t1" / "adaptive-gate-w32-t0-l0-source-L16-c0p75")
    p.add_argument("--target", default="hi")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "annotation")
    args = p.parse_args()

    pool = []
    for f in sorted(args.gen_dir.glob(f"*_to_{args.target}.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for i, r in enumerate(data["results"]):
            pool.append((f.name, i, data["source_language"], r["original_prompt"], r["output"]))
    if not pool:
        raise SystemExit(f"no *_to_{args.target}.json files in {args.gen_dir}")
    rng = random.Random(args.seed)
    picked = sorted(rng.sample(pool, min(args.n, len(pool))), key=lambda x: (x[0], x[1]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sheet = args.out_dir / f"{args.target}_ratings.csv"
    key = args.out_dir / f"{args.target}_key.json"
    with open(sheet, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "source_language", "question", "answer", "rater_1", "rater_2", "notes"])
        for k, (_, _, src, q, a) in enumerate(picked, 1):
            w.writerow([k, src, q, a, "", "", ""])
    key.write_text(json.dumps({"gen_dir": str(args.gen_dir), "target": args.target,
                               "items": [{"item_id": k, "file": f, "index": i}
                                         for k, (f, i, _, _, _) in enumerate(picked, 1)]},
                              indent=2), encoding="utf-8")
    (args.out_dir / "rating_instructions.txt").write_text(RUBRIC, encoding="utf-8")
    print(f"{len(picked)} answers from {len(pool)} in {args.gen_dir}")
    print(f"sheet -> {sheet}\nkey   -> {key}\nrubric -> {args.out_dir / 'rating_instructions.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
