"""A minimal, fully instrumented latent-predictive model over aspect tensors.

Deliberately faithful to the paper's abstraction:
    context = pooled unmasked aspects of paper p (+ optional neighbour mixer)
    target  = EMA-encoded masked aspect of paper p
    loss    = ||predictor(context, aspect_type) - sg(target)||^2   (or cos / infonce)

Every quantity the reviewer asked for is logged: per-module gradient norms,
effective rank of the predictor output, and the empirical conditional-variance
floor E||z - E[z|c]||^2.
"""
from __future__ import annotations

import copy, math
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from diag.common import set_seed, device, effective_rank


def _mlp(d_in, d_hid, d_out, depth, dropout=0.0):
    if depth <= 0:
        return nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)
    layers, d = [], d_in
    for _ in range(depth - 1):
        layers += [nn.Linear(d, d_hid), nn.GELU(), nn.Dropout(dropout)]
        d = d_hid
    layers += [nn.Linear(d, d_out)]
    return nn.Sequential(*layers)


class RefJEPA(nn.Module):
    def __init__(self, d_in, d=256, enc_depth=2, pred_depth=2, pred_width=512,
                 n_aspects=3, dropout=0.0):
        super().__init__()
        self.enc = _mlp(d_in, d, d, max(enc_depth, 1), dropout)
        self.ctx = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Embedding(n_aspects, d)
        self.pred = _mlp(2 * d, pred_width, d, max(pred_depth, 1), dropout)
        self.d = d

    def encode(self, x):                       # (B, d_in) -> (B, d)
        return self.enc(x)

    def context(self, x_ctx, mask):
        """x_ctx: (B, A, d_in), mask: (B, A) bool, True = visible."""
        h = self.enc(x_ctx)
        w = mask.float().unsqueeze(-1)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1)
        return self.ctx(pooled)

    def predict(self, c, t_idx):
        return self.pred(torch.cat([c, self.type_emb(t_idx)], dim=-1))


def _grad_norms(model: nn.Module) -> Dict[str, float]:
    groups = {"encoder": "enc.", "context_mixer": "ctx.",
              "predictor": "pred.", "type_emb": "type_emb."}
    out = {}
    for g, pre in groups.items():
        tot = 0.0
        for n, p in model.named_parameters():
            if n.startswith(pre) and p.grad is not None:
                tot += float(p.grad.detach().pow(2).sum())
        out[g] = math.sqrt(tot)
    return out


def conditional_variance_floor(X: torch.Tensor) -> float:
    """E||z - E[z|c]||^2 in the RAW target space, per masked patch.

    Context = paper identity, so the conditional mean given the context is the
    per-paper mean of the maskable aspects. This is the Prop.1 floor.
    """
    Z = F.normalize(X.float(), dim=-1)                  # (n, A, d)
    mu = Z.mean(1, keepdim=True)
    return float(((Z - mu) ** 2).sum(-1).mean())


def train_reference_jepa(X: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 0)); set_seed(seed)
    dev = device()
    X = X.float().to(dev)                                # (n, A, d_in)
    n, A, d_in = X.shape

    epochs   = int(cfg.get("epochs", 100))
    bs       = int(cfg.get("batch_size", 256))
    lr       = float(cfg.get("lr", 1e-3))
    wd       = float(cfg.get("weight_decay", 1e-4))
    loss_kind= cfg.get("loss", "l2")                     # l2 | cos | infonce
    ema      = float(cfg.get("ema", 0.996))
    tau      = float(cfg.get("temperature", 0.1))

    model = RefJEPA(d_in, d=int(cfg.get("d", 256)),
                    enc_depth=int(cfg.get("enc_depth", 2)),
                    pred_depth=int(cfg.get("pred_depth", 2)),
                    pred_width=int(cfg.get("pred_width", 512)),
                    n_aspects=A, dropout=float(cfg.get("dropout", 0.0))).to(dev)
    target = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    loss_curve: List[float] = []
    grad_log: Dict[str, List[float]] = {}
    rank_curve: List[float] = []
    upd_ratio: List[float] = []

    def _rank_probe():
        with torch.no_grad():
            idx = torch.arange(min(n, 2048), device=dev)
            m = torch.ones(len(idx), A, dtype=torch.bool, device=dev); m[:, 0] = False
            c = model.context(X[idx], m)
            p = model.predict(c, torch.zeros(len(idx), dtype=torch.long, device=dev))
        return effective_rank(p.cpu())

    rank_curve.append(_rank_probe())                     # epoch 0, untrained

    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot, nb, gsum = 0.0, 0, {}
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            xb = X[b]
            t_idx = torch.randint(0, A, (len(b),), device=dev)
            mask = torch.ones(len(b), A, dtype=torch.bool, device=dev)
            mask.scatter_(1, t_idx[:, None], False)

            c = model.context(xb, mask)
            pred = model.predict(c, t_idx)
            with torch.no_grad():
                z = target.encode(xb.gather(
                    1, t_idx[:, None, None].expand(-1, 1, d_in)).squeeze(1))

            if loss_kind == "l2":
                loss = F.mse_loss(pred, z, reduction="none").sum(-1).mean()
            elif loss_kind == "cos":
                loss = (1 - F.cosine_similarity(pred, z, dim=-1)).mean()
            else:                                        # infonce
                p_, z_ = F.normalize(pred, dim=-1), F.normalize(z, dim=-1)
                logits = p_ @ z_.T / tau
                loss = F.cross_entropy(logits,
                                       torch.arange(len(b), device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = _grad_norms(model)
            for k, v in gn.items():
                gsum[k] = gsum.get(k, 0.0) + v
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(cfg.get("clip", 5.0)))
            opt.step()
            with torch.no_grad():
                for tp, sp in zip(target.parameters(), model.parameters()):
                    tp.mul_(ema).add_(sp, alpha=1 - ema)
            tot += float(loss); nb += 1
        sched.step()
        loss_curve.append(tot / max(nb, 1))
        for k, v in gsum.items():
            grad_log.setdefault(k, []).append(v / max(nb, 1))
        wnorm = math.sqrt(sum(float(p.detach().pow(2).sum())
                              for p in model.pred.parameters()))
        upd_ratio.append(grad_log["predictor"][-1] * lr / (wnorm + 1e-12))
        if (ep + 1) % max(epochs // 5, 1) == 0:
            rank_curve.append(_rank_probe())
            print(f"    ep {ep+1:4d}  loss {loss_curve[-1]:.5f}  "
                  f"|g_pred| {grad_log['predictor'][-1]:.3e}  "
                  f"rank {rank_curve[-1]:.1f}", flush=True)

    # ---- Protocol R banks -------------------------------------------------
    model.eval()
    with torch.no_grad():
        t0 = torch.zeros(n, dtype=torch.long, device=dev)
        m0 = torch.ones(n, A, dtype=torch.bool, device=dev); m0[:, 0] = False
        Q = model.predict(model.context(X, m0), t0).cpu()
        C = target.encode(X[:, 0, :]).cpu()
    gold = torch.arange(n)

    return {"loss_curve": loss_curve, "grad_log": grad_log,
            "rank_curve": rank_curve, "update_ratio": upd_ratio,
            "Q": Q, "C": C, "gold": gold,
            "floor": conditional_variance_floor(X.cpu()),
            "final_loss": loss_curve[-1] if loss_curve else float("nan"),
            "model": model, "target": target}