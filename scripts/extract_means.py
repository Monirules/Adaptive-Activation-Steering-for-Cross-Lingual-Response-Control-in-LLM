#!/usr/bin/env python3

import argparse
import os
import time
from pathlib import Path

import torch

import steer_lib as S


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--langs", nargs="+", default=S.TRACK_LANGS)
    p.add_argument("--train-dir", type=Path,
                   default=Path(os.environ.get("CLAS_DATA", "data")) / "train")
    p.add_argument("--out-root", type=Path, default=Path(os.environ.get("CLAS_VECTORS", "vectors")))
    p.add_argument("--max-sents", type=int, default=None, help="default: all 2009")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    model, tok = S.load_model(args.model)
    tok.padding_side = "right"
    n_layers = len(S.get_layers(model))
    out = args.out_root / args.model.split("/")[-1] / "means.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"model {args.model}: {n_layers} layers, hidden {model.config.hidden_size}")

    means, norms, counts = {}, {}, {}
    for lang in args.langs:
        t0 = time.time()
        sents = [l.strip() for l in (args.train_dir / f"{lang}.txt").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        if args.max_sents:
            sents = sents[:args.max_sents]
        total = torch.zeros(n_layers, model.config.hidden_size, dtype=torch.float64, device=model.device)
        norm_sum = torch.zeros(n_layers, dtype=torch.float64, device=model.device)
        n_tok = 0
        for i in range(0, len(sents), args.batch_size):
            enc = tok(sents[i:i + args.batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_len).to(model.device)
            with torch.no_grad():
                base = getattr(model, "model", model)  # decoder only: no vocabulary logits
                hs = base(**enc, output_hidden_states=True).hidden_states[1:]  # skip embeddings
            mask = enc["attention_mask"].clone()
            if tok.bos_token_id is not None:
                mask[enc["input_ids"] == tok.bos_token_id] = 0
            m = mask[..., None].to(torch.float32)
            n_tok += int(mask.sum())
            for li, h in enumerate(hs):
                h = h.float()
                total[li] += (h * m).sum(dim=(0, 1)).double()
                norm_sum[li] += (h.norm(dim=-1) * mask).sum().double()
        means[lang] = (total / n_tok).float().cpu()
        norms[lang] = (norm_sum / n_tok).float().cpu()
        counts[lang] = n_tok
        print(f"  {lang}: {len(sents)} sentences, {n_tok} tokens, "
              f"norm L0={norms[lang][0]:.1f} Lmid={norms[lang][n_layers // 2]:.1f} "
              f"Llast={norms[lang][-1]:.1f}  ({time.time() - t0:.0f}s)", flush=True)

    torch.save({"model": args.model, "num_layers": n_layers, "langs": args.langs,
                "means": means, "norm": norms, "n_tokens": counts}, out)

    # quick sanity print: how far apart are the language means at a few layers?
    print("\nlength of DiffMean vectors (target minus others), relative to token norm:")
    for li in sorted({n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 2}):
        rel = []
        for lang in args.langs:
            v = S.steering_vector(means, li, None, lang, "others", args.langs)
            rel.append(float(v.norm() / norms[lang][li]))
        print(f"  layer {li:2d}: mean {sum(rel) / len(rel):.3f}  min {min(rel):.3f}  max {max(rel):.3f}")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
