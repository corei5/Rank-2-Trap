"""Metrics, statistics and IO for the rebuttal diagnostics.

Design rule: this module has NO dependency on the Graph-JEPA repo, so every
number here can be recomputed by a reviewer from the cached tensors alone.
"""
from __future__ import annotations

import json, math, os, random, time, contextlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, Sequence, Optional

import numpy as np
import torch


# ----------------------------------------------------------------------------- env
def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextlib.contextmanager
def timer(name: str, sink: Optional[dict] = None):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"  [time] {name}: {dt:.1f}s", flush=True)
    if sink is not None:
        sink[f"seconds/{name}"] = dt


# ----------------------------------------------------------------------------- IO
class ResultsDB:
    """Append-only JSON store. Every stage merges into one file per corpus."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.data: Dict[str, Any] = {}
        if os.path.isfile(path):
            with open(path) as f:
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError:
                    print(f"  [warn] {path} unreadable; starting fresh")

    def put(self, key: str, value: Any) -> None:
        self.data[key] = _jsonable(value)
        self.flush()

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.data and self.data[key].get("status") == "ok"

    def flush(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


# ------------------------------------------------------------------- ranking core
@torch.no_grad()
def ranks_from_banks(Q: torch.Tensor, C: torch.Tensor, gold: torch.Tensor,
                     chunk: int = 256, normalise: bool = True) -> torch.Tensor:
    """Cosine ranks of the gold candidate. Ties counted as half (unbiased).

    Q: (nq, d) queries, C: (nc, d) candidates, gold: (nq,) index into C.
    """
    dev = Q.device
    if normalise:
        Q = torch.nn.functional.normalize(Q.float(), dim=-1)
        C = torch.nn.functional.normalize(C.float(), dim=-1)
    out = torch.empty(Q.shape[0], dtype=torch.float64, device=dev)
    for i in range(0, Q.shape[0], chunk):
        q = Q[i:i + chunk]
        s = q @ C.T                                   # (b, nc)
        g = s.gather(1, gold[i:i + chunk, None])      # (b, 1)
        greater = (s > g).sum(1).double()
        ties = (s == g).sum(1).double() - 1.0         # exclude gold itself
        out[i:i + chunk] = 1.0 + greater + 0.5 * ties.clamp(min=0)
    return out


def mrr(ranks: torch.Tensor) -> float:
    return float((1.0 / ranks.double()).mean())


def recall_at(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= k).double().mean())


def bits_recovered(ranks: torch.Tensor, n_candidates: int) -> Dict[str, float]:
    """The measure the paper itself recommends (§8.4). Replaces '4693x'.

    bits = log2(N) - E[log2 rank].  Chance => 0 bits (up to O(1/n) bias).
    """
    lo = float(torch.log2(ranks.double()).mean())
    total = math.log2(n_candidates)
    # exact chance expectation for a uniform rank over N candidates
    r = torch.arange(1, n_candidates + 1, dtype=torch.float64)
    chance = float(torch.log2(r).mean())
    return {
        "bits_total": total,
        "bits_recovered": chance - lo,          # 0 at chance, log2(N) at perfect
        "bits_recovered_naive": total - lo,     # for reference only
        "mean_log2_rank": lo,
        "chance_mean_log2_rank": chance,
    }


def bootstrap_ci(x: Sequence[float], B: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(B, a.size))
    m = a[idx].mean(1)
    return {"mean": float(a.mean()),
            "lo": float(np.quantile(m, alpha / 2)),
            "hi": float(np.quantile(m, 1 - alpha / 2)),
            "n": int(a.size)}


@torch.no_grad()
def permutation_p(Q: torch.Tensor, C: torch.Tensor, gold: torch.Tensor,
                  n_perm: int = 200, seed: int = 0) -> Dict[str, float]:
    """Two-sided-ish test: P(MRR_null >= MRR_obs) under gold-label permutation."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    obs = mrr(ranks_from_banks(Q, C, gold))
    null = []
    for _ in range(n_perm):
        perm = gold[torch.randperm(gold.numel(), generator=g).to(gold.device)]
        null.append(mrr(ranks_from_banks(Q, C, perm)))
    null = np.asarray(null)
    p = float(((null >= obs).sum() + 1) / (n_perm + 1))
    return {"mrr": obs, "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)), "p_value": p,
            "z": float((obs - null.mean()) / (null.std(ddof=1) + 1e-12)),
            "n_perm": n_perm}


def effective_rank(X: torch.Tensor, eps: float = 1e-12) -> float:
    """Roy & Vetterli entropy-based effective rank."""
    Xc = X.float() - X.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc.double())
    p = s / (s.sum() + eps)
    p = p[p > eps]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def spectral_decay_kappa(X: torch.Tensor) -> float:
    """Power-law exponent kappa of the covariance spectrum (least squares on
    log-log). This is the 'anisotropy' knob used by the phase simulation."""
    Xc = X.float() - X.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc.double()) ** 2
    s = s[s > s.max() * 1e-10]
    k = min(len(s), 256)
    i = torch.arange(1, k + 1, dtype=torch.float64)
    y, x = torch.log(s[:k]), torch.log(i)
    xm, ym = x.mean(), y.mean()
    return float(-((x - xm) * (y - ym)).sum() / (((x - xm) ** 2).sum() + 1e-12))


def variance_split(X: torch.Tensor, group: torch.Tensor) -> Dict[str, float]:
    """rho = between-group variance share.  group: (n,) integer labels."""
    X = X.float()
    mu = X.mean(0, keepdim=True)
    total = float(((X - mu) ** 2).sum(1).mean())
    G = int(group.max()) + 1
    sums = torch.zeros(G, X.shape[1], device=X.device).index_add_(0, group, X)
    cnt = torch.zeros(G, device=X.device).index_add_(
        0, group, torch.ones_like(group, dtype=torch.float))
    gm = sums / cnt.clamp(min=1)[:, None]
    within = float(((X - gm[group]) ** 2).sum(1).mean())
    between = max(total - within, 0.0)
    return {"rho": between / (total + 1e-12), "total": total,
            "within": within, "between": between, "n_groups": G}


@dataclass
class StageResult:
    status: str
    payload: Dict[str, Any]
    seconds: float = 0.0
    note: str = ""

    def as_dict(self):
        d = asdict(self)
        d.update(d.pop("payload"))
        return d