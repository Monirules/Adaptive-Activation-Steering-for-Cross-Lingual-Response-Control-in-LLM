#!/usr/bin/env python3


import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

HIGH = ["ar", "de", "en", "es", "fr", "ja", "zh", "ru", "hi", "ko"]
LOW = ["cs", "el", "fa", "vi", "uk", "mt", "no", "sw", "th", "tr"]
LANGS = HIGH + LOW
CATEGORY_KEYS = ("category", "categories", "question_category", "type",
                 "question_type", "task", "task_type", "domain")
REPO_ROOT = Path(__file__).resolve().parents[1]


def read_lines(path: Path) -> list:
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def find_categories(repo: str, n: int):
    """Return (categories list or None, column used or None, columns seen)."""
    try:
        from datasets import load_dataset
    except ImportError:
        return None, None, []
    ds = None
    for split in ("en", "all", "train"):
        try:
            ds = load_dataset(repo, split=split)
            break
        except Exception:
            continue
    if ds is None:
        return None, None, []
    cols = list(ds.column_names)
    if "language_code" in cols:
        ds = ds.filter(lambda r: r["language_code"] == "en")
    key = next((k for k in CATEGORY_KEYS if k in cols), None)
    if key is None or "question_id" not in cols:
        return None, None, cols
    by_id = {}
    for r in sorted(ds, key=lambda r: r["question_id"]):
        by_id.setdefault(r["question_id"], str(r[key]))
    cats = list(by_id.values())
    if len(cats) != n:
        return None, None, cols
    return cats, key, cols


def stratified_dev(cats: list, n_dev: int, rng: random.Random) -> list:
    groups = defaultdict(list)
    for i, c in enumerate(cats):
        groups[c].append(i)
    n = len(cats)
    quota = {c: n_dev * len(ix) / n for c, ix in groups.items()}
    take = {c: int(q) for c, q in quota.items()}
    rest = n_dev - sum(take.values())
    for c in sorted(quota, key=lambda c: quota[c] - take[c], reverse=True)[:rest]:
        take[c] += 1
    dev = []
    for c in sorted(groups):
        ix = groups[c][:]
        rng.shuffle(ix)
        dev += ix[:take[c]]
    return sorted(dev)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path,
                   default=Path(os.environ.get("CLAS_DATA", "data")))
    p.add_argument("--split-file", type=Path, default=REPO_ROOT / "splits" / "split.json")
    p.add_argument("--n-dev", type=int, default=20)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--clas-repo", default="DGurgurov/CLaS-Bench")
    p.add_argument("--no-hf", action="store_true", help="skip the category lookup")
    p.add_argument("--check", action="store_true",
                   help="only verify the files against an existing split.json")
    p.add_argument("--force", action="store_true", help="overwrite an existing split.json")
    args = p.parse_args()

    val = args.data_dir / "validation"
    questions = {}
    for code in LANGS:
        path = val / f"{code}.txt"
        if not path.exists():
            sys.exit(f"missing {path}. Run scripts/download_all.sh first.")
        questions[code] = read_lines(path)
    counts = {c: len(q) for c, q in questions.items()}
    n = counts["en"]
    bad = {c: k for c, k in counts.items() if k != n}
    if bad:
        sys.exit(f"languages disagree on question count (en has {n}): {bad}")
    print(f"{len(LANGS)} languages, {n} questions each")

    if args.check:
        split = json.loads(args.split_file.read_text(encoding="utf-8"))
        ok = True
        for name in ("dev", "test"):
            ids = split[f"{name}_ids"]
            for code in LANGS:
                f = args.data_dir / f"validation_{name}" / f"{code}.txt"
                want = [questions[code][i] for i in ids]
                if not f.exists() or read_lines(f) != want:
                    print(f"MISMATCH {f}")
                    ok = False
        print("check PASS" if ok else "check FAIL")
        return 0 if ok else 1

    if args.split_file.exists() and not args.force:
        split = json.loads(args.split_file.read_text(encoding="utf-8"))
        print(f"using existing {args.split_file} (seed {split['seed']}); pass --force to remake")
        dev, test = split["dev_ids"], split["test_ids"]
    else:
        rng = random.Random(args.seed)
        cats, key, cols = (None, None, []) if args.no_hf else find_categories(args.clas_repo, n)
        if cats:
            dev = stratified_dev(cats, args.n_dev, rng)
            how = f"stratified by '{key}'"
        else:
            dev = sorted(rng.sample(range(n), args.n_dev))
            how = "random (no category column found)"
        test = [i for i in range(n) if i not in set(dev)]
        cat_counts = None
        if cats:
            cat_counts = {
                "dev": {c: sum(cats[i] == c for i in dev) for c in sorted(set(cats))},
                "test": {c: sum(cats[i] == c for i in test) for c in sorted(set(cats))},
            }
        en_hash = hashlib.sha256("\n".join(questions["en"]).encode("utf-8")).hexdigest()[:16]
        split = {
            "seed": args.seed, "n_questions": n, "n_dev": len(dev), "n_test": len(test),
            "method": how, "hf_columns_seen": cols, "category_counts": cat_counts,
            "en_questions_sha256": en_hash, "dev_ids": dev, "test_ids": test,
            "note": "Indices are 0-based line numbers in data/validation/<code>.txt. "
                    "In validation_dev/ and validation_test/, question_idx j means dev_ids[j] "
                    "or test_ids[j].",
        }
        args.split_file.parent.mkdir(parents=True, exist_ok=True)
        args.split_file.write_text(json.dumps(split, indent=2), encoding="utf-8")
        print(f"split: {how}")
        print(f"wrote {args.split_file}")

    assert not set(dev) & set(test) and len(dev) + len(test) == n
    for name, ids in (("dev", dev), ("test", test)):
        out = args.data_dir / f"validation_{name}"
        out.mkdir(parents=True, exist_ok=True)
        for code in LANGS:
            lines = [questions[code][i] for i in ids]
            (out / f"{code}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{name}: {len(ids)} questions x {len(LANGS)} languages -> {out}")
    print("dev ids:", dev)
    print("\nexample (en, first dev question):", questions["en"][dev[0]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
