import torch
import torch.nn.functional as F

import steer_lib as S


class AdaptiveSteerer:

    def __init__(self, model, layer: int, mu_all: torch.Tensor, mode: str = "gate",
                 warmup: int = 8, tau: float = 0.0, late: float = 0.0, ema: float = 0.8):
        if mode not in ("gate", "decay"):
            raise ValueError(mode)
        self.layer = S.get_layers(model)[layer]
        self.device = next(self.layer.parameters()).device
        self.mu_all = mu_all.float().to(self.device)
        self.mode, self.warmup, self.tau, self.late, self.ema_a = mode, warmup, tau, late, ema
        self.vecs = None
        self.handle = self.layer.register_forward_hook(self._hook)
        self._reset()

    def _reset(self):
        self.step = 0
        self.ema = None
        self.flags = []

    def set_batch(self, items: list):
        self.vecs = torch.stack([it["vec"] for it in items]).float().to(self.device)
        self.mu_t = torch.stack([it["mu_t"] for it in items]).float().to(self.device)
        self.mu_s = torch.stack([it["mu_s"] for it in items]).float().to(self.device)
        self._reset()

    def _scale(self) -> torch.Tensor:
        b = self.vecs.size(0)
        if self.step <= self.warmup:
            return torch.ones(b, device=self.device)
        if self.mode == "decay":
            return torch.full((b,), float(self.late), device=self.device)
        c = self.ema - self.mu_all
        drift = (F.cosine_similarity(c, self.mu_s - self.mu_all, dim=-1)
                 - F.cosine_similarity(c, self.mu_t - self.mu_all, dim=-1))
        on = (drift > -self.tau).float()
        return on + (1.0 - on) * float(self.late)

    def _hook(self, module, inputs, output):
        if self.vecs is None:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        if hs.dim() != 3 or hs.size(0) != self.vecs.size(0):
            raise RuntimeError(f"steering shape mismatch: hidden {tuple(hs.shape)} vecs {tuple(self.vecs.shape)}")
        if hs.size(1) > 1:
            self._reset()
            add = self.vecs[:, None, :]
        else:
            self.step += 1
            h = hs[:, -1, :].float()
            self.ema = h if self.ema is None else self.ema_a * self.ema + (1.0 - self.ema_a) * h
            scale = self._scale()
            self.flags.append(scale.detach().cpu())
            add = (scale[:, None] * self.vecs)[:, None, :]
        hs = hs + add.to(device=hs.device, dtype=hs.dtype)
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    def row_stats(self, new_tokens: torch.Tensor, stop_ids: set) -> list:
        if not self.flags:
            return [1.0] * new_tokens.size(0)
        fl = torch.stack(self.flags)
        out = []
        for j in range(new_tokens.size(0)):
            row = new_tokens[j].tolist()
            n = next((k + 1 for k, t in enumerate(row) if t in stop_ids), len(row))
            steps = max(min(n - 1, fl.size(0)), 1)
            out.append(round(float(fl[:steps, j].mean()), 4))
        return out

    def remove(self):
        self.handle.remove()
