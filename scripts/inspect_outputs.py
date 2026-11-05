#!/usr/bin/env python3


import argparse
import importlib.util
from collections import Counter
from pathlib import Path

import steer_lib as S


def repetitive(text: str, n: int = 8) -> bool:
    words = text.split()
    if len(words) < 4 * n:
        return False
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n)]
    return Counter(grams).most_common(1)[0][1] >= 4


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gen-dir", type=Path, required=True)
    p.add_argument("--show", nargs="*", default=[], help="pairs to print, e.g. ar_to_en")
    p.add_argument("--n", type=int, default=2, help="examples per shown pair")
    p.add_argument("--chars", type=int, default=400)
    args = p.parse_args()

    S.import_official()
    spec = importlib.util.spec_from_file_location("ev", S.official_dir() / "evaluate.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    from clas.io_utils import read_json

    lid, mapping = E.load_lid(), E.L.flores_to_code()
    by_target = {}
    total, rep, long_ = 0, 0, 0
    for f in sorted(args.gen_dir.glob("*_to_*.json")):
        d = read_json(f)
        src, tgt = d["source_language"], d["target_language"]
        preds = [E.predict_lang(lid, r["output"], mapping, 0.9) for r in d["results"]]
        c = by_target.setdefault(tgt, Counter())
        for pr in preds:
            c["TARGET" if pr == tgt else ("SOURCE" if pr == src else pr)] += 1
        for r in d["results"]:
            total += 1
            rep += repetitive(r["output"])
            long_ += len(r["output"]) > 3500
        name = f"{src}_to_{tgt}"
        if name in args.show:
            print(f"\n================ {name} ================")
            for r, pr in list(zip(d["results"], preds))[:args.n]:
                text = " ".join(r["output"].split())[:args.chars]
                print(f"[q{r['question_idx']}] question: {r['original_prompt'][:150]}")
                print(f"   detected: {pr}")
                print(f"   answer:   {text}\n")

    print(f"\nDetected answer language per target ({args.gen_dir}):")
    print("TARGET = correct language, SOURCE = answered in the question's language, unk = low confidence")
    for tgt in sorted(by_target):
        c = by_target[tgt]
        n = sum(c.values())
        top = ", ".join(f"{k} {v / n:.0%}" for k, v in c.most_common(4))
        print(f"  -> {tgt}: {top}")
    print(f"\n{total} answers: {rep} look repetitive ({rep / total:.1%}), "
          f"{long_} longer than 3500 characters ({long_ / total:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
