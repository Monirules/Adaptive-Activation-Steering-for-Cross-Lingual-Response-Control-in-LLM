

import os
import sys
from pathlib import Path

import torch

HIGH = ["ar", "de", "en", "es", "fr", "ja", "zh", "ru", "hi", "ko"]
LOW = ["cs", "el", "fa", "vi", "uk", "mt", "no", "sw", "th", "tr"]
TRACK_LANGS = HIGH + LOW


def official_dir() -> Path:
    return Path(os.environ.get("CLAS_OFFICIAL", Path(__file__).resolve().parents[1] / "official"))


def import_official():
    """Make the official `clas` package importable and return its languages module."""
    d = str(official_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    from clas import languages as L  # noqa: E402
    return L


def load_model(name: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, device_map="cuda" if torch.cuda.is_available() else None,
        trust_remote_code=True, attn_implementation="sdpa")
    model.eval()
    return model, tok


def get_layers(model):
    """Return the list of decoder layers for Llama, Cohere/Aya, Qwen, etc."""
    for path in ("model.layers", "model.model.layers", "model.language_model.layers",
                 "model.language_model.model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not find decoder layers")


class Steerer:
    """Forward hook that adds `vecs` ([batch, hidden]) to one layer's output."""

    def __init__(self, model, layer: int):
        self.layer = get_layers(model)[layer]
        self.vecs = None
        self.handle = self.layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.vecs is None:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        v = self.vecs.to(device=hs.device, dtype=hs.dtype)
        if hs.dim() == 3 and v.size(0) == hs.size(0):
            hs = hs + v[:, None, :]
        else:  # unexpected shape: refuse silently wrong steering
            raise RuntimeError(f"steering shape mismatch: hidden {tuple(hs.shape)} vecs {tuple(v.shape)}")
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    def remove(self):
        self.handle.remove()


def chat_inputs(tok, questions: list):
    """Chat-template the plain questions (no language instruction)."""
    texts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                     add_generation_prompt=True) for q in questions]
    return texts


@torch.no_grad()
def generate(model, tok, items: list, max_new_tokens: int, batch_size: int,
             steerer=None, log_every: int = 1):
    """items: list of dicts with 'text' (chat-templated) and optional 'vec' (tensor [hidden]).
    Returns list of decoded outputs in the same order."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["text"]))
    outs = [None] * len(items)
    n_batches = (len(order) + batch_size - 1) // batch_size
    for b in range(n_batches):
        idx = order[b * batch_size:(b + 1) * batch_size]
        enc = tok([items[i]["text"] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        enc.pop("token_type_ids", None)
        if steerer is not None:
            if hasattr(steerer, "set_batch"):
                steerer.set_batch([items[i] for i in idx])
            else:
                steerer.vecs = torch.stack([items[i]["vec"] for i in idx])
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
        if steerer is not None:
            steerer.vecs = None
        new = gen[:, enc["input_ids"].shape[1]:]
        stats = steerer.row_stats(new, stop_ids(model, tok)) if hasattr(steerer, "row_stats") else None
        for j, i in enumerate(idx):
            outs[i] = tok.decode(new[j], skip_special_tokens=True).strip()
            if stats is not None:
                items[i]["steer_scale"] = stats[j]
        if log_every and (b + 1) % log_every == 0:
            print(f"    batch {b + 1}/{n_batches}", flush=True)
    return outs


def stop_ids(model, tok) -> set:
    ids = set()
    eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif eos is not None:
        ids.update(eos)
    for t in (tok.eos_token_id, tok.pad_token_id):
        if t is not None:
            ids.add(t)
    return ids


def load_means(path: Path):
    d = torch.load(path, map_location="cpu")
    return d


def steering_vector(means: dict, layer: int, src: str, tgt: str, kind: str,
                    ref_langs: list) -> torch.Tensor:
    """DiffMean vector at one layer.
    kind='source': mean(target) - mean(source)            (CLaS-Bench code)
    kind='others': mean(target) - mean(all other ref langs) (our proposal)"""
    mu_t = means[tgt][layer].float()
    if kind == "source":
        mu_r = means[src][layer].float()
    elif kind == "others":
        mu_r = torch.stack([means[l][layer].float() for l in ref_langs if l != tgt]).mean(0)
    else:
        raise ValueError(kind)
    return mu_t - mu_r
