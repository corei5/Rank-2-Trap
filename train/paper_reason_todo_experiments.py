"""
paper_reason_todo_experiments.py   —  v3.1 (bugfix + rho redesign)
================================================================================
CHANGES vs v3
------------------------------------------------------------------------------
FIX  crash   `_strip_arrays` assumed str keys; stage `rho` returns int keys
             ({1: None, 4: None}) -> AttributeError on k.startswith. Now guards
             isinstance(k, str) and coerces numpy scalar keys for json.dump.

FIX  rho     v3's sweep queried with the TRUE target vector, so it measured
             candidate-bank discriminability and returned MRR=1.0 for every rho.
             It could not, even in principle, exhibit predictor collapse.
             v3.1 splits it into two sub-experiments:
               (a) geometry control  -- the old sweep, retained and RELABELLED.
                   Establishes that high rho does NOT impair bank separability.
               (b) trained sweep     -- generates
                       z_{p,a} = sqrt(rho)*alpha_a + sqrt(1-rho)*delta_{p,a}
                   with delta linearly decodable from a paper-identity code h_p
                   at signal-to-noise `snr`, then
                     * fits a closed-form RIDGE predictor  (no budget -> ceiling)
                     * TRAINS an MLP for a FIXED step budget (the JEPA analogue)
                   and runs Protocol R on both. The gap between them is the
                   severity law: identity is present and linearly recoverable,
                   but a budgeted optimiser will not spend gradient on 0.39% of
                   the loss. This mirrors oracle=0.97 vs trained=0.0002 exactly.
             RHO_GRID now contains 0.9961 == the measured corpus operating point.

FIX  misc    - duplicated STAGE_ORDER / parse_args removed
             - _infonce_fullpool no longer mutates `logits` in place
             - bits_carried guards chance_mrr(0)
             - save_artifacts tolerates ragged / object arrays
             - stage_latent best-probe selection is nan-safe

Canonical stage order:
  rankinit protocolr peraspect centering encoder latent negatives oracle fix
  poolsize rho

Usage
-----
    # the two decisive stages that were missing from the last run
    python -m train.paper_reason_todo_experiments \
        --resume --stages latent,negatives --seeds 0,1,2

    # the corrected severity law
    python -m train.paper_reason_todo_experiments --resume --stages rho

    # everything
    python -m train.paper_reason_todo_experiments --stages all --text-controls

    # re-render figures only
    python -m train.paper_reason_todo_experiments --figures-only
================================================================================
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import json
import copy
import glob
import math
import time
import random
import argparse
import platform
import importlib
import contextlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from core.data_utils.paper_graph import build_hetero_graph


# ============================================================================
#  0. DEFENSIVE IMPORTS OF THE EXISTING (FROZEN) CORE
# ============================================================================
def _imp(*candidates):
    last = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as e:                              # noqa: BLE001
            last = e
    raise ImportError(f"none of {candidates} importable: {last}")


G  = _imp("train.paper_reason_gjepa_old_1", "train.paper_reason_gjepa")
G2 = _imp("train.paper_reason_gjepa_old_2", "train.paper_reason_gjepa_v2",
          "train.paper_reason_gjepa")

DEVICE, HIDDEN, LATENT = G.DEVICE, G.HIDDEN, G.LATENT
EPOCHS_DEF, LR, WD     = G.EPOCHS, G.LR, G.WD
SEEDS_DEF              = list(G.SEEDS)
EMA_BASE, EMA_FINAL    = G.EMA_BASE, G.EMA_FINAL
RAW_DIR, CACHE_PATH    = G.RAW_DIR, G.CACHE_PATH
RWSE_CACHE, CKPT_DIR   = G.RWSE_CACHE, G.CKPT_DIR

eff_rank_only         = G.eff_rank_only
collapse_stats        = G.collapse_stats
build_patch_index     = G.build_patch_index
aspect_presence       = G.aspect_presence
build_reasoning_label = G.build_reasoning_label
eval_probe            = G.eval_probe
mean_std              = G.mean_std
verdict               = G.verdict

GraphJEPAMulti         = G2.GraphJEPAMulti
whiten_features        = G2.whiten_features
build_hetero_neighbors = G2.build_hetero_neighbors
REL_POOL               = getattr(G2, "REL_POOL", "relhetero")
REL_MAX_NEIGH          = getattr(G2, "REL_MAX_NEIGH", 8)


# ============================================================================
#  1. CONFIG
# ============================================================================
OUT_DIR = os.path.join(CKPT_DIR, "todo_experiments")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# ---- Protocol R ----
N_QUERIES   = 4000
QUERY_SEED  = 0
SCORE_CHUNK = 256
N_PERM      = 1000
BOOT_N      = 500
DEFAULT_TRANSFORMS   = ("raw", "center")
CENTERING_TRANSFORMS = ("raw", "center", "rm_top1", "rm_top2", "zca")

# ---- losses ----
INFONCE_TAU   = 0.07
INFONCE_BATCH = 1024
VIC_STD_W     = 25.0
VIC_COV_W     = 1.0

# ---- negative-sampling regimes (stage `negatives`) -------------------------
NEG_MODES  = ["cos", "inbatch_random", "aspect_matched", "aspect_hard", "fullpool"]
NEG_POOL   = 4096
NEG_HARD_K = 64

# ---- information ladder (stage `latent`) -----------------------------------
LATENT_TAPS    = ("node_pooled", "ctx_summary", "pred", "pred_no_pe", "pe_only")
RIDGE_LAMBDA   = 1e-2
RIDGE_FIT_FRAC = 0.5

# ---- pool-size scaling (stage `poolsize`) ----------------------------------
POOLSIZE_GRID = [10, 32, 100, 316, 1000, 3162, 10000, 31623]
POOLSIZE_REPS = 8

# ---- self-contained encoder ------------------------------------------------
ENC_HIDDEN   = min(HIDDEN, 128)
ENC_VARIANTS = [                    # (tag, depth, residual, mlp_in)
    ("linear_d0",  0, False, False),
    ("mlp_d0",     0, False, True),
    ("gnn_d1",     1, False, True),
    ("gnn_d2",     2, False, True),
    ("gnn_d3",     3, False, True),
    ("gnn_d2_res", 2, True,  True),
]
ENC_MAX_SEEDS = 3

# ---- tracking / artifacts --------------------------------------------------
TRACK_EVERY = 5
TRACK_ROWS  = 8000
SPEC_ROWS   = 8000
PCA_ROWS    = 4000
RANK_KEEP   = 4000

# ---- text controls ---------------------------------------------------------
TEXT_FIELD_MAP = {
    "paper":  ["title", "executive_summary", "abstract", "summary", "paper_title"],
    "claim":  ["claim", "claim_text", "claims", "statement", "text", "assertion"],
    "method": ["methodological_details", "method", "methods", "methodology",
               "approach", "method_text"],
    "result": ["key_results", "results", "result", "findings", "outcome",
               "result_text"],
}
TEXT_MAX_CHARS = 4000
NGRAM_N        = 5
SBERT_NAME     = "all-MiniLM-L6-v2"

# ---- synthetic severity-law sweep (stage `rho`) ----------------------------
# rho = fraction of TARGET variance that is between-CATEGORY (aspect).
# 0.9961 is the value measured on the real corpus (99.61% aspect / 0.39% paper).
RHO_GRID     = [0.0, 0.5, 0.9, 0.99, 0.9961, 0.999, 0.9999]
RHO_M_GRID   = [1, 4]          # members per patch -> pooling noise 1/m
RHO_N        = 20000           # synthetic papers
RHO_DIM      = 384             # target dimensionality
RHO_ID_DIM   = 64              # paper-identity code dimensionality
RHO_ASPECTS  = 3
RHO_SNR      = 1.0             # how well the context determines the residual
RHO_QUERIES  = 2000
# --- budgeted predictor (the JEPA analogue) ---
RHO_STEPS    = 2000
RHO_BATCH    = 512
RHO_HIDDEN   = 256
RHO_LR       = 1e-3
RHO_FIT_FRAC = 0.5             # ridge fit split (disjoint from queries)

# ---- cost ------------------------------------------------------------------
NUM_GPUS_BILLED     = 1
GPU_HOURLY_RATE_USD = 1.00
COST_CURRENCY       = "USD"

ART = {"spectra": {}, "pca": {}, "ranks": {}, "history": {}, "scatter": []}


# ============================================================================
#  2. UTILITIES
# ============================================================================
def _gpu_peak_gb():
    return torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0


def _pick_device(prefer_cuda=True):
    return DEVICE if (prefer_cuda and torch.cuda.is_available()) else torch.device("cpu")


class Timer(contextlib.AbstractContextManager):
    def __init__(self, name, sink):
        self.name, self.sink = name, sink

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time(); return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt, peak = time.time() - self.t0, _gpu_peak_gb()
        self.sink.append({"phase": self.name, "seconds": dt, "peak_gpu_gb": peak})
        print(f"  [time] {self.name:<44s} {dt:8.1f}s ({dt/60:5.1f} min) peak={peak:5.2f} GB",
              flush=True)
        return False


def _fmt_hms(sec):
    sec = int(round(sec)); return f"{sec//3600:d}h{(sec%3600)//60:02d}m{sec%60:02d}s"


def _fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "---"
        return f"{x:.{nd}f}"
    except Exception:                                       # noqa: BLE001
        return "---"


def set_all_seeds(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _subsample(X, n, seed=0):
    if X.size(0) <= n:
        return X
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.size(0), generator=g)[:n]
    return X[idx.to(X.device)]


def _mm(v):
    v = [x for x in v if x is not None]
    if not v:
        return [float("nan"), float("nan")]
    return [float(np.nanmean(v)), float(np.nanstd(v))]


def spectrum_of(X, rows=SPEC_ROWS):
    Xs = _subsample(X.detach().float().cpu(), rows)
    Xc = Xs - Xs.mean(0, keepdim=True)
    try:
        s = torch.linalg.svdvals(Xc)
    except Exception:                                       # noqa: BLE001
        return None, float("nan")
    s = s[s > 1e-12]
    if s.numel() == 0:
        return None, 0.0
    p = (s / s.sum()).numpy()
    er = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    return p, er


def pca2_of(X, rows=PCA_ROWS):
    Xs = _subsample(X.detach().float().cpu(), rows)
    Xc = Xs - Xs.mean(0, keepdim=True)
    try:
        U, S, _ = torch.pca_lowrank(Xc, q=2, center=False)
        return (U[:, :2] * S[:2]).numpy()
    except Exception:                                       # noqa: BLE001
        return None


def dc_stats(X, rows=SPEC_ROWS):
    """
    Shared-mean dominance -- the quantity effective rank is blind to.
      dc_ratio  = ||mu|| / E||x - mu||
      dc_energy = ||mu||^2 / (||mu||^2 + E||x - mu||^2)
    Large dc_ratio => <mu, delta_c> (query-independent) swamps <delta_q, delta_c>.
    """
    Xs = _subsample(X.detach().float().cpu(), rows)
    mu = Xs.mean(0)
    res = Xs - mu
    mn = float(mu.norm())
    rn = float(res.norm(dim=1).mean())
    e_res = float(res.pow(2).sum(1).mean())
    return {"mean_norm": mn, "resid_norm": rn,
            "dc_ratio": mn / max(rn, 1e-12),
            "dc_energy": (mn ** 2) / max(mn ** 2 + e_res, 1e-12)}


def bootstrap_mrr_ci(ranks, n_boot=BOOT_N, alpha=0.05, seed=0):
    r = np.asarray(ranks, dtype=np.float64)
    if r.size < 2:
        return float("nan"), float("nan"), float("nan")
    rr = 1.0 / r
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, rr.size, size=(n_boot, rr.size))
    bs = rr[idx].mean(1)
    return (float(rr.mean()), float(np.quantile(bs, alpha / 2)),
            float(np.quantile(bs, 1 - alpha / 2)))


def ridge_fit(X, Y, lam=RIDGE_LAMBDA):
    X = X.float(); Y = Y.float()
    bx, by = X.mean(0, keepdim=True), Y.mean(0, keepdim=True)
    Xc, Yc = X - bx, Y - by
    d = Xc.size(1)
    A = Xc.T @ Xc
    A = A + lam * (torch.trace(A) / max(d, 1)) * torch.eye(d, device=A.device, dtype=A.dtype)
    try:
        W = torch.linalg.solve(A, Xc.T @ Yc)
    except Exception:                                       # noqa: BLE001
        W = torch.linalg.lstsq(A, Xc.T @ Yc).solution
    return W, bx, by


def ridge_apply(X, W, bx, by):
    return (X.float() - bx) @ W + by


def variance_decomposition(vectors, group_ids):
    """
    With group = aspect, `within` is the between-PAPER variance retrieval needs,
    and `between` is what the JEPA objective can trivially exploit.
    """
    X = vectors.detach().float().cpu()
    g = group_ids.detach().cpu().long()
    m = X.mean(0, keepdim=True)
    tot = float((X - m).pow(2).sum(1).mean())
    n = X.size(0)
    betw = within = 0.0
    for gid in torch.unique(g):
        sel = (g == gid)
        Xa = X[sel]
        if Xa.numel() == 0:
            continue
        ma = Xa.mean(0, keepdim=True)
        betw += float(sel.sum()) * float((ma - m).pow(2).sum())
        within += float((Xa - ma).pow(2).sum(1).sum())
    betw /= max(n, 1); within /= max(n, 1)
    s = max(betw + within, 1e-12)
    return {"total_var": tot, "between_group_var": betw, "within_group_var": within,
            "between_frac": betw / s, "within_frac": within / s}


# ============================================================================
#  3. PATCH MEMBERSHIP + DEGREE
# ============================================================================
def build_membership(data, active):
    out = {}
    for a in active:
        if a == "paper":
            idx = torch.arange(data["paper"].num_nodes)
            out[a] = (idx.clone(), idx.clone()); continue
        chosen = None
        for et in data.edge_index_dict:
            src, rel, dst = et
            if src == "paper" and dst == a and (rel.startswith("has") or rel == a):
                chosen = et; break
        if chosen is None:
            for et in data.edge_index_dict:
                src, rel, dst = et
                if src == "paper" and dst == a and not rel.startswith("rev_"):
                    chosen = et; break
        if chosen is None:
            raise RuntimeError(f"no paper->{a} edge type in {list(data.edge_index_dict)}")
        ei = data.edge_index_dict[chosen]
        out[a] = (ei[0].detach().cpu().clone(), ei[1].detach().cpu().clone())
        print(f"  [membership] {a:<8s} via {chosen} | E={ei.size(1)}")
    return out


def membership_counts(membership, active, P):
    cnt = {}
    for a in active:
        pid, _ = membership[a]
        c = torch.zeros(P, dtype=torch.long).index_add_(0, pid, torch.ones_like(pid))
        cnt[a] = c
    return cnt


def node_type_degrees(data):
    deg = {}
    for nt in data.node_types:
        n = data[nt].num_nodes
        d = torch.zeros(n)
        for et, ei in data.edge_index_dict.items():
            if et[2] != nt:
                continue
            dst = ei[1].detach().cpu()
            d.index_add_(0, dst, torch.ones(dst.numel()))
        deg[nt] = {"mean_in_degree": float(d.mean()), "n_nodes": int(n)}
    return deg


# ============================================================================
#  4. POOLERS
# ============================================================================
class SetPooler(nn.Module):
    def __init__(self, mode, dim):
        super().__init__()
        self.mode = mode
        if mode == "deepsets":
            self.phi = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            self.rho = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                                     nn.Linear(dim, dim))
        elif mode == "attn":
            self.V = nn.Linear(dim, dim)
            self.w = nn.Linear(dim, 1, bias=False)
            self.rho = nn.LayerNorm(dim)

    @staticmethod
    def _ssum(src, index, P):
        out = torch.zeros(P, src.size(1), device=src.device, dtype=src.dtype)
        return out.index_add_(0, index, src)

    @staticmethod
    def _scnt(index, P, device):
        c = torch.zeros(P, device=device).index_add_(
            0, index, torch.ones(index.numel(), device=device))
        return c.clamp_min(1.0).unsqueeze(1)

    def _ssoftmax(self, scores, index, P):
        mx = torch.full((P,), -1e30, device=scores.device)
        mx = mx.scatter_reduce(0, index, scores, reduce="amax", include_self=True)
        ex = torch.exp(scores - mx[index])
        den = torch.zeros(P, device=scores.device).index_add_(0, index, ex)
        return ex / den[index].clamp_min(1e-12)

    def forward(self, z, paper_ids, node_ids, P):
        src = z[node_ids]
        if self.mode == "mean":
            return self._ssum(src, paper_ids, P) / self._scnt(paper_ids, P, src.device)
        if self.mode == "sum":
            return self._ssum(src, paper_ids, P)
        if self.mode == "deepsets":
            return self.rho(self._ssum(self.phi(src), paper_ids, P))
        if self.mode == "attn":
            s = self.w(torch.tanh(self.V(src))).squeeze(-1)
            a = self._ssoftmax(s, paper_ids, P).unsqueeze(-1)
            return self.rho(self._ssum(src * a, paper_ids, P))
        raise ValueError(self.mode)


def _ema_(tgt, src, m):
    if tgt is None:
        return
    with torch.no_grad():
        for pt, ps in zip(tgt.parameters(), src.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for bt, bs in zip(tgt.buffers(), src.buffers()):
            bt.copy_(bs)


# ============================================================================
#  5. SELF-CONTAINED ENCODER + JEPA HEAD
# ============================================================================
def _etkey(et):
    return "__".join(et)


class HeteroEncoder(nn.Module):
    """depth 0 = no message passing; depth L = L typed mean-aggregation rounds."""
    def __init__(self, in_dims, edge_types, hidden, out_dim, depth=2,
                 residual=False, mlp_in=True, norm=True):
        super().__init__()
        self.depth, self.residual = depth, residual
        self.ntypes = list(in_dims)
        self.etypes = [et for et in edge_types if et[0] in in_dims and et[2] in in_dims]
        self.inp = nn.ModuleDict()
        for nt, d in in_dims.items():
            self.inp[nt] = (nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                          nn.Linear(hidden, hidden))
                            if mlp_in else nn.Linear(d, hidden))
        self.msg, self.upd, self.nrm = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(depth):
            self.msg.append(nn.ModuleDict(
                {_etkey(et): nn.Linear(hidden, hidden, bias=False) for et in self.etypes}))
            self.upd.append(nn.ModuleDict(
                {nt: nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU())
                 for nt in self.ntypes}))
            self.nrm.append(nn.ModuleDict(
                {nt: (nn.LayerNorm(hidden) if norm else nn.Identity())
                 for nt in self.ntypes}))
        self.out = nn.ModuleDict({nt: nn.Linear(hidden, out_dim) for nt in self.ntypes})

    def forward(self, x_dict, edge_index_dict):
        h = {nt: self.inp[nt](x_dict[nt].float()) for nt in self.ntypes}
        h0 = h
        for l in range(self.depth):
            agg = {nt: torch.zeros_like(h[nt]) for nt in self.ntypes}
            cnt = {nt: torch.zeros(h[nt].size(0), 1, device=h[nt].device)
                   for nt in self.ntypes}
            for et in self.etypes:
                ei = edge_index_dict.get(et)
                if ei is None:
                    continue
                s, d = ei[0], ei[1]
                agg[et[2]].index_add_(0, d, self.msg[l][_etkey(et)](h[et[0]][s]))
                cnt[et[2]].index_add_(0, d, torch.ones(d.numel(), 1, device=d.device))
            h = {nt: self.nrm[l][nt](self.upd[l][nt](
                    torch.cat([h[nt], agg[nt] / cnt[nt].clamp_min(1.0)], -1)))
                 for nt in self.ntypes}
        if self.residual and self.depth > 0:
            h = {nt: h[nt] + h0[nt] for nt in self.ntypes}
        return {nt: self.out[nt](h[nt]) for nt in self.ntypes}


class JEPAHead(nn.Module):
    """Attribute names match GraphJEPAMulti so protocol_r_model works unchanged."""
    def __init__(self, dim, pe_dim, nhead=4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, nhead, 2 * dim, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.ctx_mixer = nn.TransformerEncoder(layer, num_layers=1)
        self.pe_proj   = nn.Linear(pe_dim, dim)
        self.predictor = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim),
                                       nn.GELU(), nn.Linear(2 * dim, dim))


# ============================================================================
#  5b. LOSSES
# ============================================================================
def _cos_loss(pred, tgt):
    pn, tn = F.normalize(pred, dim=-1), F.normalize(tgt, dim=-1)
    return (1.0 - (pn * tn).sum(-1)).mean()


def _infonce_inbatch(pred, tgt, tau=INFONCE_TAU):
    pn, tn = F.normalize(pred, dim=-1), F.normalize(tgt, dim=-1)
    b = pn.size(0)
    logits = pn @ tn.T / tau
    lab = torch.arange(b, device=pn.device)
    return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))


def _infonce_fullpool(pred, tgt, neg, neg_ids, rows, tau=INFONCE_TAU):
    """Positives in-batch + negatives drawn from the real full candidate pool."""
    pn, tn = F.normalize(pred, dim=-1), F.normalize(tgt, dim=-1)
    nn_ = F.normalize(neg, dim=-1)
    b = pn.size(0)
    logits = torch.cat([pn @ tn.T, pn @ nn_.T], 1) / tau
    dup = (neg_ids.view(1, -1) == rows.view(-1, 1))          # accidental positives
    if dup.any():                                            # FIX(v3.1): no in-place
        full = torch.zeros_like(logits, dtype=torch.bool)
        full[:, b:] = dup
        logits = logits.masked_fill(full, -1e4)
    lab = torch.arange(b, device=pn.device)
    return F.cross_entropy(logits, lab)


def _infonce_hard(pred, tgt, pool, pool_ids, rows, k=NEG_HARD_K, tau=INFONCE_TAU):
    """Top-k hardest negatives mined per row out of a uniform candidate sample."""
    pn, tn = F.normalize(pred, dim=-1), F.normalize(tgt, dim=-1)
    cn = F.normalize(pool, dim=-1)
    sims = pn @ cn.T
    dup = (pool_ids.view(1, -1) == rows.view(-1, 1))
    sims = sims.masked_fill(dup, -1e4)
    kk = min(k, sims.size(1))
    hard_idx = sims.topk(kk, dim=1).indices
    hard = cn[hard_idx]
    pos = (pn * tn).sum(-1, keepdim=True)
    negs = torch.einsum("bl,bkl->bk", pn, hard)
    logits = torch.cat([pos, negs], 1) / tau
    return F.cross_entropy(logits, torch.zeros(pn.size(0), dtype=torch.long,
                                               device=pn.device))


def _vicreg_target(tgt):
    t_c = tgt - tgt.mean(0, keepdim=True)
    l = VIC_STD_W * F.relu(1.0 - torch.sqrt(t_c.var(0) + 1e-4)).mean()
    cov = (t_c.T @ t_c) / max(1, tgt.size(0) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return l + VIC_COV_W * off.pow(2).sum() / tgt.size(1)


def _jepa_loss(pred, tgt, loss_mode, tgt_vic=False):
    if loss_mode == "cos":
        loss = _cos_loss(pred, tgt)
    elif loss_mode == "infonce":
        loss = _infonce_inbatch(pred, tgt)
    else:
        raise ValueError(loss_mode)
    if tgt_vic:
        loss = loss + _vicreg_target(tgt)
    return loss


# ============================================================================
#  6. TRAINERS
# ============================================================================
def _sample_rows(gen, pres_d, maskable_idx, A, same_aspect, batch):
    """
    Returns (rows, target_aspect, present_mask, aspect_or_None).
    Same-aspect batches remove the aspect shortcut: every in-batch negative is
    then a different PAPER, not a different aspect.
    """
    if same_aspect:
        ai = int(torch.randint(0, A, (1,), generator=gen, device=DEVICE).item())
        elig = maskable_idx[pres_d[maskable_idx][:, ai]]
        if elig.numel() > 0:
            if elig.numel() > batch:
                sel = torch.randperm(elig.numel(), generator=gen, device=DEVICE)[:batch]
                rows = elig[sel]
            else:
                rows = elig
            ta = torch.full((rows.numel(),), ai, device=DEVICE, dtype=torch.long)
            return rows, ta, pres_d[rows], ai
    pres_m = pres_d[maskable_idx]
    Pm = maskable_idx.size(0)
    rnd = torch.rand(Pm, A, generator=gen, device=DEVICE); rnd[~pres_m] = -1.0
    tgt_aspect = rnd.argmax(1)
    if Pm > batch:
        sel = torch.randperm(Pm, generator=gen, device=DEVICE)[:batch]
    else:
        sel = torch.arange(Pm, device=DEVICE)
    return maskable_idx[sel], tgt_aspect[sel], pres_m[sel], None


def train_variant(seed, data, active, membership, pres, rwse, maskable, pe_dim,
                  pool_mode="mean", loss_mode="cos", whitened_x=None,
                  epochs=None, tgt_vic=False, track=False,
                  neg_mode=None, neg_pool=NEG_POOL, neg_hard_k=NEG_HARD_K):
    """
    Frozen GraphJEPAMulti encoder + our pooling/loss.
    `neg_mode` (when given) overrides `loss_mode` and selects one of NEG_MODES.
    """
    epochs = epochs or EPOCHS_DEF
    set_all_seeds(seed)
    full = data.to(DEVICE)          # HeteroData.to is IN PLACE
    P, A = full["paper"].num_nodes, len(active)
    pe = {a: rwse[a].to(DEVICE) for a in active}
    pres_d, maskable_idx = pres.to(DEVICE), maskable.to(DEVICE)
    mem = {a: (membership[a][0].to(DEVICE), membership[a][1].to(DEVICE)) for a in active}

    model = GraphJEPAMulti(data.metadata(), HIDDEN, LATENT, active, pe_dim,
                           "mean", mah_k=0, rel_neighbors=None).to(DEVICE)
    pooler = SetPooler(pool_mode, LATENT).to(DEVICE)
    pooler_t = copy.deepcopy(pooler).to(DEVICE) if any(
        p.requires_grad for p in pooler.parameters()) else None
    if pooler_t is not None:
        for p in pooler_t.parameters():
            p.requires_grad_(False)

    x_in = full.x_dict
    if whitened_x is not None:
        x_in = {k: (whitened_x[k].to(DEVICE) if k in whitened_x else full.x_dict[k])
                for k in full.x_dict}

    with torch.no_grad():
        model.encode_nodes(x_in, full.edge_index_dict)
        model.encode_nodes_tgt(x_in, full.edge_index_dict)
    model.init_target(); model.freeze_target()

    params = [p for p in model.parameters() if p.requires_grad] + \
             [p for p in pooler.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WD)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    pe_stack = torch.stack([pe[a] for a in active], 1)

    same_aspect = neg_mode in ("aspect_matched", "aspect_hard", "fullpool")
    cand_by_aspect = ({i: torch.nonzero(pres_d[:, i], as_tuple=False).squeeze(1)
                       for i in range(A)}
                      if neg_mode in ("aspect_hard", "fullpool") else {})

    def pool_all(z, use_t=False):
        pl = pooler_t if (use_t and pooler_t is not None) else pooler
        return torch.stack([pl(z[a], mem[a][0], mem[a][1], P) for a in active], 1)

    history = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    for ep in range(epochs):
        model.train(); pooler.train()
        m = EMA_FINAL - (EMA_FINAL - EMA_BASE) * (
            0.5 * (1 + np.cos(np.pi * ep / max(1, epochs - 1))))
        with torch.no_grad():
            zt = model.encode_nodes_tgt(x_in, full.edge_index_dict)
            tgt_emb = pool_all({a: zt[a].detach() for a in active}, use_t=True)
        zc = model.encode_nodes(x_in, full.edge_index_dict)
        ctx_emb = pool_all(zc)

        use_batch = (neg_mode not in (None, "cos")) or (loss_mode == "infonce")
        batch = INFONCE_BATCH if use_batch else maskable_idx.numel()
        rows, ta, pm, ai = _sample_rows(gen, pres_d, maskable_idx, A, same_aspect, batch)
        b = rows.numel()

        ctx_mask = pm.clone(); ctx_mask[torch.arange(b, device=DEVICE), ta] = False
        mixed = model.ctx_mixer(ctx_emb[rows], src_key_padding_mask=~ctx_mask)
        w = ctx_mask.float().unsqueeze(-1)
        ctx_summary = (mixed * w).sum(1) / w.sum(1).clamp_min(1.0)
        tgt_pe = pe_stack[rows][torch.arange(b, device=DEVICE), ta]
        pred = model.predictor(ctx_summary + model.pe_proj(tgt_pe))
        tgt = tgt_emb[rows][torch.arange(b, device=DEVICE), ta]

        if neg_mode in (None, "cos"):
            loss = _jepa_loss(pred, tgt, loss_mode, tgt_vic=tgt_vic)
        elif neg_mode in ("inbatch_random", "aspect_matched"):
            loss = _infonce_inbatch(pred, tgt)
        elif neg_mode in ("fullpool", "aspect_hard"):
            if ai is None:
                loss = _infonce_inbatch(pred, tgt)
            else:
                cid = cand_by_aspect[ai]
                nn_ = min(neg_pool, cid.numel())
                nid = cid[torch.randint(0, cid.numel(), (nn_,), generator=gen,
                                        device=DEVICE)]
                if neg_mode == "fullpool":
                    loss = _infonce_fullpool(pred, tgt, tgt_emb[nid, ai], nid, rows)
                else:
                    loss = _infonce_hard(pred, tgt, tgt_emb[nid, ai], nid, rows,
                                         k=neg_hard_k)
        else:
            raise ValueError(neg_mode)

        opt.zero_grad(); loss.backward(); opt.step()
        model.ema(m); _ema_(pooler_t, pooler, m)

        if track and (ep % TRACK_EVERY == 0 or ep == epochs - 1):
            with torch.no_grad():
                flat = tgt_emb[maskable_idx][pres_d[maskable_idx]]
                er_p = float(eff_rank_only(_subsample(flat, TRACK_ROWS)))
                er_n = float(np.nanmean([eff_rank_only(_subsample(zt[a], TRACK_ROWS))
                                         for a in active]))
            history.append([ep, float(loss.item()), er_p, er_n])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_sec = time.time() - t0

    model.eval(); pooler.eval()
    t1 = time.time()
    with torch.no_grad():
        zt = model.encode_nodes_tgt(x_in, full.edge_index_dict)
        patch_repr = pool_all({a: zt[a] for a in active}, use_t=True)
        node_rk = {a: float(eff_rank_only(zt[a])) for a in active}
        pool_rk = {a: float(eff_rank_only(patch_repr[pres_d[:, i], i]))
                   for i, a in enumerate(active)}
        pm_flat = patch_repr[maskable_idx][pres_d[maskable_idx]]
        gid = torch.nonzero(pres_d[maskable_idx], as_tuple=False)[:, 1]
        n_keep = min(pm_flat.size(0), SPEC_ROWS)
        diag = {"node_eff_rank_per_aspect": node_rk,
                "node_eff_rank_mean": float(np.nanmean(list(node_rk.values()))),
                "pool_eff_rank_per_aspect": pool_rk,
                "patchmean_eff_rank": float(eff_rank_only(pm_flat)),
                "pooled_dc": dc_stats(pm_flat),
                "pooled_variance": variance_decomposition(pm_flat[:n_keep],
                                                          gid[:n_keep]),
                "node_dc_per_aspect": {a: dc_stats(_subsample(zt[a], SPEC_ROWS))
                                       for a in active},
                "node_latents_ref": zt, "pooled_flat_ref": pm_flat}
    timings = {"train_sec": train_sec, "diag_sec": time.time() - t1,
               "peak_gpu_gb": _gpu_peak_gb()}
    return model, pooler, patch_repr, pe, pres_d, maskable_idx, diag, timings, history


def train_encoder_variant(seed, ctx, depth=2, residual=False, mlp_in=True,
                          pool_mode="mean", loss_mode="cos", whitened_x=None,
                          epochs=None, hidden=ENC_HIDDEN, track=False):
    """Same JEPA dynamics, but with OUR encoder so depth/residual are variables."""
    epochs = epochs or EPOCHS_DEF
    set_all_seeds(seed)
    data, active, membership = ctx["data"], ctx["active"], ctx["membership"]
    full = data.to(DEVICE)
    P, A = full["paper"].num_nodes, len(active)
    pe = {a: ctx["rwse"][a].to(DEVICE) for a in active}
    pres_d = ctx["pres"].to(DEVICE); maskable_idx = ctx["maskable"].to(DEVICE)
    mem = {a: (membership[a][0].to(DEVICE), membership[a][1].to(DEVICE)) for a in active}

    x_in = {k: v for k, v in full.x_dict.items()}
    if whitened_x is not None:
        for k in list(x_in):
            if k in whitened_x:
                x_in[k] = whitened_x[k].to(DEVICE)
    in_dims = {nt: int(x_in[nt].size(1)) for nt in x_in}

    enc = HeteroEncoder(in_dims, list(full.edge_index_dict.keys()), hidden, LATENT,
                        depth=depth, residual=residual, mlp_in=mlp_in).to(DEVICE)
    pooler = SetPooler(pool_mode, LATENT).to(DEVICE)
    head = JEPAHead(LATENT, ctx["pe_dim"]).to(DEVICE)
    enc_t = copy.deepcopy(enc).to(DEVICE)
    pooler_t = copy.deepcopy(pooler).to(DEVICE) if any(
        p.requires_grad for p in pooler.parameters()) else None
    for mod in (enc_t, pooler_t):
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad_(False)

    params = list(enc.parameters()) + list(pooler.parameters()) + list(head.parameters())
    opt = torch.optim.Adam([p for p in params if p.requires_grad], lr=LR, weight_decay=WD)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    pe_stack = torch.stack([pe[a] for a in active], 1)

    def pool_all(z, use_t=False):
        pl = pooler_t if (use_t and pooler_t is not None) else pooler
        return torch.stack([pl(z[a], mem[a][0], mem[a][1], P) for a in active], 1)

    history = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for ep in range(epochs):
        enc.train(); pooler.train(); head.train()
        m = EMA_FINAL - (EMA_FINAL - EMA_BASE) * (
            0.5 * (1 + np.cos(np.pi * ep / max(1, epochs - 1))))
        with torch.no_grad():
            zt = enc_t(x_in, full.edge_index_dict)
            tgt_emb = pool_all({a: zt[a].detach() for a in active}, use_t=True)
        zc = enc(x_in, full.edge_index_dict)
        ctx_emb = pool_all(zc)

        rows, ta, pm, _ = _sample_rows(gen, pres_d, maskable_idx, A, False, INFONCE_BATCH)
        b = rows.numel()
        ctx_mask = pm.clone(); ctx_mask[torch.arange(b, device=DEVICE), ta] = False
        mixed = head.ctx_mixer(ctx_emb[rows], src_key_padding_mask=~ctx_mask)
        w = ctx_mask.float().unsqueeze(-1)
        summ = (mixed * w).sum(1) / w.sum(1).clamp_min(1.0)
        tgt_pe = pe_stack[rows][torch.arange(b, device=DEVICE), ta]
        pred = head.predictor(summ + head.pe_proj(tgt_pe))
        tgt = tgt_emb[rows][torch.arange(b, device=DEVICE), ta]
        loss = _jepa_loss(pred, tgt, loss_mode)

        opt.zero_grad(); loss.backward(); opt.step()
        _ema_(enc_t, enc, m); _ema_(pooler_t, pooler, m)

        if track and (ep % TRACK_EVERY == 0 or ep == epochs - 1):
            with torch.no_grad():
                flat = tgt_emb[maskable_idx][pres_d[maskable_idx]]
                er_p = float(eff_rank_only(_subsample(flat, TRACK_ROWS)))
                er_n = float(np.nanmean([eff_rank_only(_subsample(zt[a], TRACK_ROWS))
                                         for a in active]))
            history.append([ep, float(loss.item()), er_p, er_n])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_sec = time.time() - t0

    enc.eval(); pooler.eval(); head.eval()
    with torch.no_grad():
        zt = enc_t(x_in, full.edge_index_dict)
        patch_repr = pool_all({a: zt[a] for a in active}, use_t=True)
        node_rk = {a: float(eff_rank_only(zt[a])) for a in active}
        pool_rk = {a: float(eff_rank_only(patch_repr[pres_d[:, i], i]))
                   for i, a in enumerate(active)}
        pm_flat = patch_repr[maskable_idx][pres_d[maskable_idx]]
        diag = {"node_eff_rank_per_aspect": node_rk,
                "node_eff_rank_mean": float(np.nanmean(list(node_rk.values()))),
                "pool_eff_rank_per_aspect": pool_rk,
                "patchmean_eff_rank": float(eff_rank_only(pm_flat)),
                "pooled_dc": dc_stats(pm_flat),
                "node_latents_ref": zt, "pooled_flat_ref": pm_flat}
    tm = {"train_sec": train_sec, "peak_gpu_gb": _gpu_peak_gb()}
    del enc, enc_t, pooler, pooler_t
    return head, patch_repr, pe, pres_d, maskable_idx, diag, tm, history


@torch.no_grad()
def untrained_encoder_ranks(ctx, depth, residual=False, mlp_in=True,
                            hidden=ENC_HIDDEN, seed=0, whitened=False):
    set_all_seeds(seed)
    full = ctx["data"].to(DEVICE)
    x_in = {k: v for k, v in full.x_dict.items()}
    if whitened:
        for k in list(x_in):
            if k in ctx["whitened_x"]:
                x_in[k] = ctx["whitened_x"][k].to(DEVICE)
    in_dims = {nt: int(x_in[nt].size(1)) for nt in x_in}
    enc = HeteroEncoder(in_dims, list(full.edge_index_dict.keys()), hidden, LATENT,
                        depth=depth, residual=residual, mlp_in=mlp_in).to(DEVICE).eval()
    z = enc(x_in, full.edge_index_dict)
    out = {}
    for nt in z:
        zz = _subsample(z[nt], SPEC_ROWS)
        out[nt] = {"eff_rank": float(eff_rank_only(zz)), **dc_stats(zz)}
    del enc, z
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ============================================================================
#  7. PROTOCOL R
# ============================================================================
def build_query_set(pres, maskable, active, n_queries=N_QUERIES, seed=QUERY_SEED):
    g = torch.Generator().manual_seed(seed)
    m = maskable.cpu()
    q = m[torch.randperm(m.numel(), generator=g)[:min(n_queries, m.numel())]]
    pres_q = pres.cpu()[q]
    rnd = torch.rand(q.numel(), len(active), generator=g); rnd[~pres_q] = -1.0
    ta = rnd.argmax(1)
    out = {a: q[ta == i] for i, a in enumerate(active)}
    print("  [protocolR] queries: " + " ".join(f"{a}={out[a].numel()}" for a in active) +
          f" | total={sum(out[a].numel() for a in active)}")
    return out


def chance_mrr(n):
    n = int(n)
    if n < 1:
        return float("nan")
    h = (math.log(n) + 0.5772156649 + 1.0 / (2 * n)) if n > 50 else \
        sum(1.0 / k for k in range(1, n + 1))
    return h / n


def _mc_null_mrr(n_cand, n_q, n_perm=N_PERM, seed=0):
    rng = np.random.default_rng(seed)
    return (1.0 / rng.integers(1, n_cand + 1, size=(n_perm, n_q))).mean(axis=1)


def fit_bank_transform(cand, mode, eps=1e-5):
    """Fit a retrieval-space transform on the CANDIDATE BANK, apply to both."""
    C = cand.detach().float()
    mu = C.mean(0, keepdim=True)
    if mode == "raw":
        return lambda X: X.detach().float()
    if mode == "center":
        return lambda X: X.detach().float() - mu
    if mode.startswith("rm_top"):
        k = max(1, int(mode.replace("rm_top", "")))
        Cc = C - mu
        try:
            q = min(k + 4, min(Cc.shape) - 1)
            _, _, V = torch.pca_lowrank(Cc, q=max(q, k), center=False)
            V = V[:, :k].contiguous()
        except Exception:                                   # noqa: BLE001
            return lambda X: X.detach().float() - mu
        def f(X, V=V, mu=mu):
            Y = X.detach().float() - mu
            return Y - (Y @ V) @ V.T
        return f
    if mode in ("zca", "whiten"):
        Cc = C - mu
        d = Cc.size(1)
        cov = (Cc.T @ Cc) / max(1, Cc.size(0) - 1)
        cov = cov + eps * torch.eye(d, device=cov.device, dtype=cov.dtype)
        try:
            ev, U = torch.linalg.eigh(cov)
            W = U @ torch.diag(ev.clamp_min(eps).rsqrt()) @ U.T
        except Exception:                                   # noqa: BLE001
            return lambda X: X.detach().float() - mu
        return lambda X, W=W, mu=mu: (X.detach().float() - mu) @ W
    raise ValueError(mode)


@torch.no_grad()
def rank_against_full_pool(query_vec, cand_vec, gold_pos, chunk=SCORE_CHUNK,
                           transform=None, want_margin=False):
    dev = cand_vec.device
    q = query_vec.to(dev).float()
    c = cand_vec.float()
    if transform is not None:
        q, c = transform(q), transform(c)
    q = F.normalize(q, dim=-1); c = F.normalize(c, dim=-1)
    gold_pos = gold_pos.to(dev).long()
    ranks = torch.empty(q.size(0), device=dev)
    margins = torch.empty(q.size(0), device=dev) if want_margin else None
    for s in range(0, q.size(0), chunk):
        e = min(s + chunk, q.size(0))
        sims = q[s:e] @ c.T
        gold = sims.gather(1, gold_pos[s:e].view(-1, 1))
        ranks[s:e] = (sims > gold).sum(1).float() + 1.0
        if want_margin:
            mu = sims.mean(1, keepdim=True)
            sd = sims.std(1, keepdim=True).clamp_min(1e-9)
            margins[s:e] = ((gold - mu) / sd).squeeze(1)
    return (ranks, margins) if want_margin else ranks


def _metrics_from_ranks(ranks, n_cand, tag, seed=0, with_ci=True):
    r = ranks.detach().cpu().numpy() if torch.is_tensor(ranks) else np.asarray(ranks)
    mrr = float((1.0 / r).mean())
    null = _mc_null_mrr(n_cand, len(r), seed=seed)
    out = {"MRR": mrr, "Hits@1": float((r <= 1).mean()), "Hits@10": float((r <= 10).mean()),
           "mean_rank": float(r.mean()), "median_rank": float(np.median(r)),
           "n_queries": int(len(r)), "n_candidates": int(n_cand),
           "chance_MRR": chance_mrr(n_cand), "null_MRR_mean": float(null.mean()),
           "p_value_vs_null": float((null >= mrr).mean()), "tag": tag}
    if with_ci:
        _, lo, hi = bootstrap_mrr_ci(r, seed=seed)
        out["MRR_ci95"] = [lo, hi]
        out["above_chance"] = bool(lo > out["chance_MRR"])
    return out


def _pool_micro(per_aspect):
    pa = {k: v for k, v in per_aspect.items()
          if k not in ("micro", "_ranks") and isinstance(v, dict) and "MRR" in v}
    if not pa:
        return {"MRR": float("nan")}
    n = sum(v["n_queries"] for v in pa.values())
    wavg = lambda k: sum(v[k] * v["n_queries"] for v in pa.values()) / max(n, 1)
    return {"MRR": wavg("MRR"), "Hits@1": wavg("Hits@1"), "Hits@10": wavg("Hits@10"),
            "chance_MRR": wavg("chance_MRR"), "n_queries": n,
            "p_value_vs_null": max(v["p_value_vs_null"] for v in pa.values()),
            "any_above_chance": any(v.get("above_chance") for v in pa.values())}


@torch.no_grad()
def protocol_r_model(model, patch_repr, pe, pres_d, active, queries, seed=0,
                     transforms=DEFAULT_TRANSFORMS):
    P, A, L = patch_repr.shape
    pe_stack = torch.stack([pe[a] for a in active], 1)
    per_t = {t: {} for t in transforms}
    ranks_t = {t: [] for t in transforms}
    bank_stats = {}
    for i, a in enumerate(active):
        q_ids = queries[a].to(DEVICE)
        if q_ids.numel() < 2:
            continue
        cand_ids = torch.nonzero(pres_d[:, i], as_tuple=False).squeeze(1)
        pos_of = torch.full((P,), -1, dtype=torch.long, device=DEVICE)
        pos_of[cand_ids] = torch.arange(cand_ids.numel(), device=DEVICE)
        q_ids = q_ids[pos_of[q_ids] >= 0]
        gold_pos = pos_of[q_ids]

        mask = pres_d[q_ids].clone(); mask[:, i] = False        # leak-safe
        mixed = model.ctx_mixer(patch_repr[q_ids], src_key_padding_mask=~mask)
        w = mask.float().unsqueeze(-1)
        summ = (mixed * w).sum(1) / w.sum(1).clamp_min(1.0)
        pred = model.predictor(summ + model.pe_proj(pe_stack[q_ids][:, i]))
        cand = patch_repr[cand_ids, i]

        qn = F.normalize(pred - pred.mean(0, keepdim=True), dim=-1)
        disp = float((qn @ qn.T).mean())                      # ~1 => identical queries
        bank_stats[a] = {
            "cand_eff_rank": float(eff_rank_only(_subsample(cand, SPEC_ROWS))),
            "cand_dc": dc_stats(cand),
            "query_eff_rank": float(eff_rank_only(_subsample(pred, SPEC_ROWS))),
            "query_dc": dc_stats(pred),
            "query_self_similarity": disp,
            "n_candidates": int(cand_ids.numel())}

        for t in transforms:
            fn = fit_bank_transform(cand, t)
            if t == "raw":
                rk, mg = rank_against_full_pool(pred, cand, gold_pos, transform=fn,
                                                want_margin=True)
                bank_stats[a]["gold_margin_z"] = float(mg.mean())
            else:
                rk = rank_against_full_pool(pred, cand, gold_pos, transform=fn)
            per_t[t][a] = _metrics_from_ranks(rk, cand_ids.numel(), f"model/{t}/{a}", seed)
            ranks_t[t].append(rk.detach().cpu().numpy())
    micro_t = {t: _pool_micro(per_t[t]) for t in transforms}
    cat = {t: (np.concatenate(v) if v else np.zeros(0)) for t, v in ranks_t.items()}
    return {"per_aspect_by_transform": per_t, "micro_by_transform": micro_t,
            "per_aspect": per_t.get("raw", {}), "micro": micro_t.get("raw", {}),
            "ranks": cat.get("raw", np.zeros(0)),
            "ranks_by_transform": cat, "bank_stats": bank_stats}


# ---------------- pool-size scaling via exact hypergeometric thinning -------
def mrr_at_pool_size(full_ranks, n_full, n, rng, reps=POOLSIZE_REPS):
    """
    Exact MRR the SAME model would obtain against a uniform random candidate
    subset of size n containing the gold item. If k items beat the gold in the
    full pool, the number beating it in the subset is
    Hypergeometric(k, n_full-1-k, n-1). No re-scoring required.
    """
    k = np.asarray(full_ranks, dtype=np.int64) - 1
    k = np.clip(k, 0, n_full - 1)
    nbad = np.maximum(n_full - 1 - k, 0)
    if n >= n_full:
        return float((1.0 / (1.0 + k)).mean())
    vals = []
    for _ in range(reps):
        beat = rng.hypergeometric(np.maximum(k, 0), nbad, n - 1)
        vals.append(float((1.0 / (1.0 + beat)).mean()))
    return float(np.mean(vals))


def bits_carried(mrr, n):
    """
    Effective pool size N_eff at which a random ranker matches this MRR;
    log2(N / N_eff) is the information the model actually supplies.
    """
    if not np.isfinite(mrr) or mrr <= 0 or n < 2:
        return 0.0
    lo, hi = 1.0, float(n)
    for _ in range(60):
        mid = math.sqrt(max(lo, 1e-9) * hi)
        c = chance_mrr(max(int(round(mid)), 1))              # FIX(v3.1): guard n=0
        if not np.isfinite(c):
            break
        if c > mrr:
            lo = mid
        else:
            hi = mid
    n_eff = 0.5 * (lo + hi)
    return float(max(0.0, math.log2(max(n / max(n_eff, 1.0), 1.0))))


# ============================================================================
#  8. ORACLE + LEAKAGE CONTROLS
# ============================================================================
def aspect_feature_matrix(data, active, membership, feats=None, device=None,
                          raw_x=None):
    dev = device if device is not None else _pick_device()
    P = int(data["paper"].num_nodes)
    mats, dims = [], []
    for a in active:
        if feats is not None and a in feats:
            x = feats[a]
        elif raw_x is not None and a in raw_x:
            x = raw_x[a]
        else:
            x = data[a].x
        x = x.detach().float().to(dev)
        dims.append(int(x.size(1)))
        pid, nid = membership[a]
        pid, nid = pid.to(dev).long(), nid.to(dev).long()
        out = torch.zeros(P, x.size(1), device=dev, dtype=x.dtype)
        out.index_add_(0, pid, x[nid])
        cnt = torch.zeros(P, device=dev, dtype=x.dtype)
        cnt.index_add_(0, pid, torch.ones(pid.numel(), device=dev, dtype=x.dtype))
        mats.append(out / cnt.clamp_min(1.0).unsqueeze(1))
    if len(set(dims)) != 1:
        raise RuntimeError(f"aspect feature dims disagree: {dict(zip(active, dims))}")
    return torch.stack(mats, 1)


@torch.no_grad()
def oracle_protocol_r(asp_feat, pres, active, queries, include_paper=None,
                      seed=0, tag="oracle", transforms=DEFAULT_TRANSFORMS):
    asp = asp_feat.to(DEVICE).float(); pres_d = pres.to(DEVICE)
    P, A, D = asp.shape
    if include_paper is not None:
        include_paper = include_paper.detach().float().to(DEVICE)
        if include_paper.size(1) != D:
            print(f"  [warn] {tag}: paper dim {include_paper.size(1)} != aspect dim {D}"
                  f" -> paper node dropped from the context")
            include_paper = None
    out, all_ranks = {}, []
    for i, a in enumerate(active):
        q_ids = queries[a].to(DEVICE)
        cand_ids = torch.nonzero(pres_d[:, i], as_tuple=False).squeeze(1)
        pos_of = torch.full((P,), -1, dtype=torch.long, device=DEVICE)
        pos_of[cand_ids] = torch.arange(cand_ids.numel(), device=DEVICE)
        q_ids = q_ids[pos_of[q_ids] >= 0]
        if q_ids.numel() < 2:
            continue
        gold_pos = pos_of[q_ids]

        mask = pres_d[q_ids].clone(); mask[:, i] = False
        w = mask.float().unsqueeze(-1)
        ctx = (asp[q_ids] * w).sum(1); den = w.sum(1).clamp_min(1.0)
        if include_paper is not None:
            ctx = ctx + include_paper[q_ids]; den = den + 1.0
        ctx = ctx / den
        cand = asp[cand_ids, i]

        ranks = rank_against_full_pool(ctx, cand, gold_pos)
        out[a] = _metrics_from_ranks(ranks, cand_ids.numel(), f"{tag}/{a}", seed)
        out[a]["dc"] = dc_stats(cand)
        out[a]["eff_rank"] = float(eff_rank_only(_subsample(cand, SPEC_ROWS)))
        for t in transforms:
            if t == "raw":
                continue
            fn = fit_bank_transform(cand, t)
            rk = rank_against_full_pool(ctx, cand, gold_pos, transform=fn)
            out[a][f"MRR_{t}"] = float((1.0 / rk.detach().cpu().numpy()).mean())
        cheat = rank_against_full_pool(asp[q_ids, i], cand, gold_pos)
        out[a]["cheat_MRR"] = float((1.0 / cheat.detach().cpu().numpy()).mean())
        all_ranks.append(ranks.detach().cpu().numpy())
    out["micro"] = _pool_micro(out)
    out["_ranks"] = np.concatenate(all_ranks) if all_ranks else np.zeros(0)
    return out


def _walk_collect(obj, keyset, acc, depth=0, max_depth=8):
    if depth > max_depth or obj is None:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in keyset:
                if isinstance(v, str) and v.strip():
                    acc.append(v.strip())
                elif isinstance(v, (list, tuple)):
                    for e in v:
                        if isinstance(e, str) and e.strip():
                            acc.append(e.strip())
                        else:
                            _walk_collect(e, keyset, acc, depth + 1, max_depth)
                elif isinstance(v, dict):
                    _walk_collect(v, keyset | {"text", "value", "content", "description"},
                                  acc, depth + 1, max_depth)
            else:
                _walk_collect(v, keyset, acc, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        for e in obj:
            _walk_collect(e, keyset, acc, depth + 1, max_depth)


def extract_text(d, keys, max_chars=TEXT_MAX_CHARS):
    acc = []
    _walk_collect(d, {str(k).lower() for k in keys}, acc)
    unpacked = []
    for s in acc:
        st = s.strip()
        if st.startswith("[") or st.startswith("{"):
            try:
                sub = json.loads(st)
                tmp = []
                _walk_collect(sub, {"description", "text", "claim", "value", "content",
                                    "statement"}, tmp)
                if tmp:
                    unpacked.extend(tmp); continue
            except Exception:                               # noqa: BLE001
                pass
        unpacked.append(s)
    seen, uniq = set(), []
    for s in unpacked:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return " ".join(uniq)[:max_chars]


def dump_json_schema(path, max_lines=60):
    try:
        d = json.load(open(path))
    except Exception as e:                                  # noqa: BLE001
        print(f"  [schema] cannot read {path}: {e}"); return
    lines = []

    def walk(o, prefix="", depth=0):
        if len(lines) >= max_lines or depth > 4:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                p = f"{prefix}.{k}" if prefix else k
                if isinstance(v, str):
                    lines.append(f"    {p:<52s} str[{len(v)}] {v[:40]!r}")
                elif isinstance(v, (int, float, bool)) or v is None:
                    lines.append(f"    {p:<52s} {type(v).__name__}")
                elif isinstance(v, list):
                    kinds = {type(e).__name__ for e in v[:5]}
                    lines.append(f"    {p:<52s} list[{len(v)}] of {sorted(kinds)}")
                    if v and isinstance(v[0], (dict, list)):
                        walk(v[0], p + "[0]", depth + 1)
                else:
                    lines.append(f"    {p:<52s} dict")
                    walk(v, p, depth + 1)
        elif isinstance(o, list) and o:
            walk(o[0], prefix + "[0]", depth + 1)

    walk(d)
    print(f"  [schema] {os.path.basename(path)}")
    print("\n".join(lines[:max_lines]))


def load_texts(raw_dir, active, limit=None, dump_schema=True):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
    if limit:
        files = files[:limit]
    if dump_schema and files:
        dump_json_schema(files[0])
    fields = sorted(set(list(active) + ["paper"]))
    texts = {k: [] for k in fields}
    bad = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:                                   # noqa: BLE001
            bad += 1
            for k in fields:
                texts[k].append("")
            continue
        for k in fields:
            texts[k].append(extract_text(d, TEXT_FIELD_MAP.get(k, [k])))
    empt = {k: sum(1 for t in texts[k] if not t) for k in fields}
    print(f"  [texts] files={len(files)} unreadable={bad} empty_per_field={empt}")
    for k in fields:
        nz = [len(t) for t in texts[k] if t]
        print(f"    {k:<8s} non-empty={len(nz)} mean_len={int(np.mean(nz)) if nz else 0}")
    if any(v > 0.5 * len(files) for v in empt.values()):
        print("  [texts][WARN] >50% empty for some field -> read the [schema] dump above.")
    return texts, files


def bm25_retrieval(query_texts, cand_texts, gold_idx, k1=1.5, b=0.75, chunk=64):
    from sklearn.feature_extraction.text import CountVectorizer
    import scipy.sparse as sp
    vec = CountVectorizer(lowercase=True, stop_words="english", min_df=2)
    Xc = vec.fit_transform(cand_texts).tocsc().astype(np.float32)
    if Xc.shape[1] == 0:
        raise RuntimeError("empty BM25 vocabulary (candidate texts are blank)")
    dl = np.asarray(Xc.sum(1)).ravel(); avgdl = max(dl.mean(), 1e-6)
    df = np.asarray((Xc > 0).sum(0)).ravel(); N = Xc.shape[0]
    idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    Xc = Xc.tocoo()
    denom = Xc.data + k1 * (1 - b + b * dl[Xc.row] / avgdl)
    w = (Xc.data * (k1 + 1) / denom) * idf[Xc.col]
    W = sp.coo_matrix((w, (Xc.row, Xc.col)), shape=(N, len(idf))).tocsr()
    Q = (vec.transform(query_texts) > 0).astype(np.float32).tocsr()
    ranks = np.empty(Q.shape[0])
    for s in range(0, Q.shape[0], chunk):
        e = min(s + chunk, Q.shape[0])
        S = (Q[s:e] @ W.T).toarray()
        gold = S[np.arange(e - s), gold_idx[s:e]][:, None]
        ranks[s:e] = (S > gold).sum(1) + 1
    return ranks


def strip_shared_ngrams(ctx_text, tgt_text, n=NGRAM_N):
    tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())
    c, t = tok(ctx_text), tok(tgt_text)
    if len(c) < n or len(t) < n:
        return ctx_text
    tset = {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}
    keep = np.ones(len(c), dtype=bool)
    for i in range(len(c) - n + 1):
        if tuple(c[i:i + n]) in tset:
            keep[i:i + n] = False
    return " ".join(w for w, k in zip(c, keep) if k)


def sbert_embed(texts, batch=256):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(SBERT_NAME, device=DEVICE)
    e = m.encode(texts, batch_size=batch, convert_to_numpy=True,
                 show_progress_bar=False, normalize_embeddings=False)
    return torch.from_numpy(e).float()


# ============================================================================
#  9. SYNTHETIC SEVERITY-LAW SWEEP  (REDESIGNED IN v3.1)
# ============================================================================
class _RhoPredictor(nn.Module):
    """The JEPA analogue: identity code + aspect code -> target embedding."""
    def __init__(self, id_dim, n_aspects, dim, hidden=RHO_HIDDEN):
        super().__init__()
        self.emb = nn.Embedding(n_aspects, hidden)
        self.net = nn.Sequential(
            nn.Linear(id_dim + hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim))

    def forward(self, h, a):
        return self.net(torch.cat([h.float(), self.emb(a)], -1))


def _rho_make_task(n, dim, id_dim, rho, m, n_aspects, snr, seed, device):
    """
    z_{p,a} = sqrt(rho) * alpha_a  +  sqrt(1-rho) * (delta_{p,a} + pooling noise)

    alpha_a          : unit-norm aspect centroid           (3 possibilities)
    delta_{p,a}      : instance residual, LINEARLY decodable from h_p at SNR snr
    pooling noise    : 1/sqrt(m) shrinkage from averaging m members

    Every component is unit-scale, so `rho` IS the between-aspect variance
    fraction -- directly comparable to the corpus measurement (0.9961).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g, device=device)

    alpha = F.normalize(rn(n_aspects, dim), dim=-1)                     # (A,D)
    h = rn(n, id_dim) / math.sqrt(id_dim)                               # (N,K)
    Wa = rn(n_aspects, id_dim, dim) / math.sqrt(id_dim)                 # (A,K,D)

    decodable = torch.einsum("nk,akd->nad", h, Wa)
    decodable = F.normalize(decodable, dim=-1)                          # unit norm
    noise = F.normalize(rn(n, n_aspects, dim), dim=-1)
    s = float(min(max(snr, 0.0), 1.0))
    delta = s * decodable + math.sqrt(max(1.0 - s * s, 0.0)) * noise

    pool_noise = F.normalize(rn(n, n_aspects, dim), dim=-1) / math.sqrt(m)
    z = (math.sqrt(rho) * alpha.unsqueeze(0)
         + math.sqrt(max(1.0 - rho, 0.0)) * (delta + pool_noise))
    return h, z


def _rho_protocol_r(pred_fn, h, z, q_idx, n_aspects, device):
    """Protocol R over the full synthetic pool, micro-averaged across aspects."""
    n = z.size(0)
    mrrs, qranks, self_sims = [], [], []
    for a in range(n_aspects):
        av = torch.full((q_idx.numel(),), a, dtype=torch.long, device=device)
        with torch.no_grad():
            qv = pred_fn(h[q_idx], av)
        rk = rank_against_full_pool(qv, z[:, a], q_idx)
        mrrs.append(float((1.0 / rk.detach().cpu().numpy()).mean()))
        qranks.append(float(eff_rank_only(_subsample(qv, SPEC_ROWS))))
        qn = F.normalize(qv - qv.mean(0, keepdim=True), dim=-1)
        self_sims.append(float((qn @ qn.T).mean()))
    return (float(np.mean(mrrs)), float(np.mean(qranks)), float(np.mean(self_sims)))


def rho_sweep_trained(n=RHO_N, dim=RHO_DIM, id_dim=RHO_ID_DIM, rhos=RHO_GRID,
                      ms=RHO_M_GRID, n_aspects=RHO_ASPECTS, snr=RHO_SNR,
                      steps=RHO_STEPS, batch=RHO_BATCH, lr=RHO_LR,
                      n_queries=RHO_QUERIES, seed=0, device=None):
    """
    THE SEVERITY LAW.

    For each rho we compare two predictors on the SAME data:
      ridge   : closed-form best linear map h -> z_a. No optimisation budget.
                Measures whether identity is RECOVERABLE at all.
      trained : an MLP trained with the cosine JEPA loss for a FIXED budget.
                Measures whether a budgeted optimiser WILL recover it.

    A widening ridge/trained gap as rho -> 1 is the mechanism: identity is
    present and linearly decodable, but worth (1-rho) of the loss, so the
    optimiser never spends gradient on it. This is the synthetic analogue of
    oracle=0.97 vs trained=0.0002.
    """
    device = device or _pick_device()
    rows = []
    for m in ms:
        for rho in rhos:
            t0 = time.time()
            h, z = _rho_make_task(n, dim, id_dim, rho, m, n_aspects, snr,
                                  seed=seed, device=device)
            gsplit = torch.Generator().manual_seed(seed + 1)
            perm = torch.randperm(n, generator=gsplit).to(device)
            n_fit = int(RHO_FIT_FRAC * n)
            fit_idx = perm[:n_fit]
            q_idx = perm[n_fit:n_fit + min(n_queries, n - n_fit)]

            # ---- variance check (should equal rho) ----
            flat = z.reshape(-1, dim)
            gid = torch.arange(n_aspects, device=device).repeat(n)
            vd = variance_decomposition(flat[:SPEC_ROWS], gid[:SPEC_ROWS])
            bank_rk = float(eff_rank_only(_subsample(z[:, 0], SPEC_ROWS)))
            bank_dc = dc_stats(z[:, 0])

            # ---- (1) ridge ceiling: no optimisation budget ----
            Ws = []
            for a in range(n_aspects):
                Ws.append(ridge_fit(h[fit_idx], z[fit_idx, a]))
            def ridge_fn(hq, av, Ws=Ws):
                out = torch.empty(hq.size(0), dim, device=device)
                for a in range(n_aspects):
                    sel = (av == a)
                    if sel.any():
                        W, bx, by = Ws[a]
                        out[sel] = ridge_apply(hq[sel], W, bx, by)
                return out
            mrr_ridge, qrk_ridge, _ = _rho_protocol_r(ridge_fn, h, z, q_idx,
                                                      n_aspects, device)

            # ---- (2) budgeted predictor: the JEPA analogue ----
            set_all_seeds(seed)
            net = _RhoPredictor(id_dim, n_aspects, dim).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            gtr = torch.Generator(device=device).manual_seed(seed + 2)
            losses = []
            for st in range(steps):
                bi = fit_idx[torch.randint(0, fit_idx.numel(), (batch,),
                                           generator=gtr, device=device)]
                ba = torch.randint(0, n_aspects, (batch,), generator=gtr, device=device)
                pred = net(h[bi], ba)
                tgt = z[bi, ba]
                loss = _cos_loss(pred, tgt)
                opt.zero_grad(); loss.backward(); opt.step()
                if st % max(1, steps // 10) == 0 or st == steps - 1:
                    losses.append(float(loss.item()))
            net.eval()
            mrr_tr, qrk_tr, self_tr = _rho_protocol_r(
                lambda hq, av: net(hq, av), h, z, q_idx, n_aspects, device)

            row = {"m": int(m), "rho": float(rho),
                   "between_frac_measured": vd["between_frac"],
                   "within_frac_measured": vd["within_frac"],
                   "bw_ratio": float(rho / max(1.0 - rho, 1e-9)),
                   "bank_eff_rank": bank_rk, "bank_dc_ratio": bank_dc["dc_ratio"],
                   "MRR_ridge": mrr_ridge, "MRR_trained": mrr_tr,
                   "query_eff_rank_trained": qrk_tr,
                   "query_eff_rank_ridge": qrk_ridge,
                   "query_self_sim_trained": self_tr,
                   "final_loss": losses[-1] if losses else float("nan"),
                   "chance_MRR": chance_mrr(n),
                   "seconds": time.time() - t0}
            rows.append(row)
            print(f"    m={m} rho={rho:<7.4f} | bank_rk={bank_rk:6.1f} "
                  f"| ridge={mrr_ridge:.4f}  trained={mrr_tr:.4f} "
                  f"(gap {mrr_ridge/max(mrr_tr,1e-9):7.1f}x) "
                  f"| q_rk={qrk_tr:6.2f} self-sim={self_tr:+.3f} "
                  f"| loss={row['final_loss']:.4f} | {row['seconds']:.0f}s", flush=True)
            del h, z, net, Ws
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # collapse frontier: first rho where trained drops below 50% of the ridge ceiling
    thr = {}
    for m in ms:
        sub = sorted([r for r in rows if r["m"] == m], key=lambda r: r["rho"])
        below = [r["rho"] for r in sub if r["MRR_trained"] < 0.5 * r["MRR_ridge"]]
        thr[f"m={m}"] = float(min(below)) if below else None
    return rows, thr


def rho_sweep_geometry(n=RHO_N, dim=RHO_DIM, rhos=RHO_GRID, ms=RHO_M_GRID, seed=0,
                       n_queries=RHO_QUERIES, n_aspects=RHO_ASPECTS):
    """
    NEGATIVE CONTROL (this was v3's only rho experiment).

    Query with a TRUE context vector rather than a learned prediction. Retrieval
    stays perfect for every rho, which establishes that a high between-category
    variance fraction does NOT, by itself, impair candidate-bank separability.
    Any collapse must therefore come from the PREDICTOR, not the geometry.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=dim).astype(np.float32); u /= np.linalg.norm(u)
    rows = []
    for m in ms:
        for rho in rhos:
            var_pool = 1.0 + 1.0 / m
            amp = math.sqrt(rho / max(1.0 - rho, 1e-9) * var_pool)
            delta = (rng.normal(size=(n, dim)) / math.sqrt(dim)).astype(np.float32)
            pooled = np.empty((n, n_aspects, dim), dtype=np.float32)
            for a in range(n_aspects):
                eta = (rng.normal(size=(n, m, dim)) / math.sqrt(dim)).astype(np.float32)
                pooled[:, a] = amp * u + delta + eta.mean(1)
            tgt = torch.from_numpy(pooled[:, 0])
            ctx = torch.from_numpy(pooled[:, 1:].mean(1))
            st = dc_stats(tgt)
            er = float(eff_rank_only(_subsample(tgt, SPEC_ROWS)))
            q = rng.choice(n, size=min(n_queries, n), replace=False)
            gold = torch.from_numpy(q.astype(np.int64))
            mrr_raw = float((1.0 / rank_against_full_pool(ctx[q], tgt, gold).numpy()).mean())
            mrr_ctr = float((1.0 / rank_against_full_pool(
                ctx[q], tgt, gold,
                transform=fit_bank_transform(tgt, "center")).numpy()).mean())
            rows.append({"m": int(m), "rho": float(rho), "amp": amp,
                         "erank_pooled": er, "dc_ratio": st["dc_ratio"],
                         "dc_energy": st["dc_energy"], "MRR_raw": mrr_raw,
                         "MRR_center": mrr_ctr, "chance_MRR": chance_mrr(n)})
            print(f"    [control] m={m} rho={rho:<7.4f} | erank={er:7.1f} "
                  f"| dc_ratio={st['dc_ratio']:8.2f} "
                  f"| MRR raw={mrr_raw:.4f} centered={mrr_ctr:.4f}")
    return rows


def measured_redundancy(ctx, res):
    out = {}
    try:
        for tag, feats in (("input_raw", None), ("input_whitened", ctx["whitened_x"])):
            asp = aspect_feature_matrix(ctx["data"], ctx["active"], ctx["membership"],
                                        feats=feats, raw_x=ctx.get("raw_x"))
            flat = asp.reshape(-1, asp.size(-1))
            out[tag] = {**dc_stats(flat),
                        "eff_rank": float(eff_rank_only(_subsample(flat, SPEC_ROWS)))}
            del asp, flat
    except Exception as e:                                  # noqa: BLE001
        print(f"  [warn] measured_redundancy(inputs) failed: {e}")
    pr = (res.get("protocolr") or {}).get("euclidean_mean_raw", {})
    if isinstance(pr, dict) and pr.get("pooled_dc"):
        out["trained_pooled"] = pr["pooled_dc"]
    if isinstance(pr, dict) and pr.get("pooled_variance"):
        out["trained_variance"] = pr["pooled_variance"]
    ce = (res.get("centering") or {}).get("bank_stats_mean")
    if ce:
        out["trained_query"] = {"dc_ratio": ce.get("query_dc_ratio"),
                                "dc_energy": ce.get("query_dc_energy")}
    for k, v in out.items():
        if "dc_ratio" not in v:
            continue
        print(f"  [redundancy] {k:<16s} dc_ratio={_fmt(v.get('dc_ratio'), 2)} "
              f"dc_energy={_fmt(v.get('dc_energy'), 4)} "
              f"eff_rank={_fmt(v.get('eff_rank'), 1)}")
    return out


def estimate_rho_real(data, active, membership, aspect="claim", max_papers=5000,
                      seed=0, raw_x=None):
    if aspect not in active:
        return float("nan")
    x = (raw_x[aspect] if (raw_x is not None and aspect in raw_x) else data[aspect].x)
    x = x.detach().float().cpu()
    pid, nid = membership[aspect]
    pid, nid = pid.cpu().tolist(), nid.cpu().tolist()
    byp = defaultdict(list)
    for p, nd in zip(pid, nid):
        byp[p].append(nd)
    keys = [k for k, v in byp.items() if len(v) >= 2]
    if not keys:
        return float("nan")
    rng = np.random.default_rng(seed); rng.shuffle(keys); keys = keys[:max_papers]
    sims = []
    for k in keys:
        v = F.normalize(x[torch.tensor(byp[k], dtype=torch.long)], dim=-1)
        S = v @ v.T
        iu = torch.triu_indices(S.size(0), S.size(0), offset=1)
        sims.append(float(S[iu[0], iu[1]].mean()))
    return float(np.mean(sims))


# ============================================================================
# 10. THE INFORMATION LADDER (stage `latent`)
# ============================================================================
@torch.no_grad()
def compute_tap(model, patch_repr, pe_stack, pres_d, ai, ids, tap="pred", chunk=8192):
    """
    Query representation at one tap of the pipeline:
      node_pooled : masked mean of the CONTEXT patch embeddings (pre-mixer)
      ctx_summary : after ctx_mixer + masked mean
      pred        : predictor(summ + pe_proj(pe))   <- what Protocol R uses
      pred_no_pe  : predictor(summ)                 <- drop the aspect code
      pe_only     : predictor(pe_proj(pe))          <- aspect code ALONE
    If `pe_only` matches `pred`, the query carries zero paper information.
    """
    outs = []
    for s in range(0, ids.numel(), chunk):
        idx = ids[s:s + chunk]
        mask = pres_d[idx].clone(); mask[:, ai] = False
        w = mask.float().unsqueeze(-1)
        if tap == "node_pooled":
            q = (patch_repr[idx] * w).sum(1) / w.sum(1).clamp_min(1.0)
        elif tap == "pe_only":
            q = model.predictor(model.pe_proj(pe_stack[idx][:, ai]))
        else:
            mixed = model.ctx_mixer(patch_repr[idx], src_key_padding_mask=~mask)
            summ = (mixed * w).sum(1) / w.sum(1).clamp_min(1.0)
            if tap == "ctx_summary":
                q = summ
            elif tap == "pred_no_pe":
                q = model.predictor(summ)
            else:
                q = model.predictor(summ + model.pe_proj(pe_stack[idx][:, ai]))
        outs.append(q)
    return torch.cat(outs, 0)


@torch.no_grad()
def information_ladder(model, patch_repr, pe, pres_d, active, queries, oracle_asp,
                       seed=0, taps=LATENT_TAPS, fit_frac=RIDGE_FIT_FRAC):
    """
    For every tap: (i) Protocol R against the LEARNED candidate bank, and
    (ii) a ridge probe from the tap onto the ORACLE target space (fit on papers
    disjoint from the query set), scored by Protocol R in that space. (ii) upper
    bounds what ANY linear read-out could extract from that tap.
    """
    P, A, L = patch_repr.shape
    pe_stack = torch.stack([pe[a] for a in active], 1)
    direct = {t: {"per_aspect": {}, "ranks": []} for t in taps}
    probe = {t: {"per_aspect": {}} for t in taps}
    geom = {t: defaultdict(list) for t in taps}
    g = torch.Generator().manual_seed(seed)

    for i, a in enumerate(active):
        q_ids = queries[a].to(DEVICE)
        if q_ids.numel() < 2:
            continue
        cand_ids = torch.nonzero(pres_d[:, i], as_tuple=False).squeeze(1)
        pos_of = torch.full((P,), -1, dtype=torch.long, device=DEVICE)
        pos_of[cand_ids] = torch.arange(cand_ids.numel(), device=DEVICE)
        q_ids = q_ids[pos_of[q_ids] >= 0]
        gold_pos = pos_of[q_ids]
        cand = patch_repr[cand_ids, i]

        perm = torch.randperm(cand_ids.numel(), generator=g).to(DEVICE)
        n_fit = max(2, int(fit_frac * cand_ids.numel()))
        fit_ids = cand_ids[perm[:n_fit]]
        qset = set(int(x) for x in q_ids.detach().cpu().tolist())
        keep = torch.tensor([int(x) not in qset for x in fit_ids.detach().cpu().tolist()],
                            device=DEVICE)
        fit_ids = fit_ids[keep]
        y_all = oracle_asp[:, i, :].to(DEVICE).float()

        for t in taps:
            qv = compute_tap(model, patch_repr, pe_stack, pres_d, i, q_ids, tap=t)
            rk = rank_against_full_pool(qv, cand, gold_pos)
            direct[t]["per_aspect"][a] = _metrics_from_ranks(
                rk, cand_ids.numel(), f"latent/{t}/{a}", seed)
            direct[t]["ranks"].append(rk.detach().cpu().numpy())
            qn = F.normalize(qv - qv.mean(0, keepdim=True), dim=-1)
            geom[t]["self_sim"].append(float((qn @ qn.T).mean()))
            geom[t]["eff_rank"].append(float(eff_rank_only(_subsample(qv, SPEC_ROWS))))
            geom[t]["dc_ratio"].append(dc_stats(qv)["dc_ratio"])
            try:
                xf = compute_tap(model, patch_repr, pe_stack, pres_d, i, fit_ids, tap=t)
                W, bx, by = ridge_fit(xf, y_all[fit_ids])
                qp = ridge_apply(qv, W, bx, by)
                rkp = rank_against_full_pool(qp, y_all[cand_ids], gold_pos)
                probe[t]["per_aspect"][a] = _metrics_from_ranks(
                    rkp, cand_ids.numel(), f"probe/{t}/{a}", seed)
                del xf, W, qp, rkp
            except Exception as e:                          # noqa: BLE001
                print(f"    [warn] ridge probe {t}/{a}: {e}")
            del qv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    res = {}
    for t in taps:
        res[t] = {"micro": _pool_micro(direct[t]["per_aspect"]),
                  "per_aspect": direct[t]["per_aspect"],
                  "probe_micro": _pool_micro(probe[t]["per_aspect"]),
                  "probe_per_aspect": probe[t]["per_aspect"],
                  "geometry": {k: float(np.mean(v)) for k, v in geom[t].items()},
                  "_ranks": (np.concatenate(direct[t]["ranks"])
                             if direct[t]["ranks"] else np.zeros(0))}
    return res


# ============================================================================
# 11. FIGURES
# ============================================================================
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F",
      "black": "#000000"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "lines.linewidth": 1.3, "lines.markersize": 4.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": "none", "legend.facecolor": "white",
        "figure.constrained_layout.use": True,
    })
    return plt


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"),
                    bbox_inches="tight", pad_inches=0.02)
    print(f"  [fig] {name}.pdf / .png")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _chance(res):
    return res.get("config", {}).get("chance_MRR")


def _oracle_level(res):
    return ((res.get("oracle") or {}).get("ctx_range") or (None, None))[0]


def _measured_bw_ratio(res):
    """Between/within variance ratio measured on the real corpus."""
    pv = ((res.get("protocolr") or {}).get("euclidean_mean_raw") or {}).get("pooled_variance")
    if not pv:
        return None
    w = pv.get("within_frac")
    b = pv.get("between_frac")
    if not w or not b:
        return None
    return float(b / max(w, 1e-12))


# ------------------------------------------------------------------- fig 1
def fig_rank_vs_mrr(res):
    pts = ART["scatter"]
    if not pts:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    styles = {"baseline": (CB["red"], "o"), "fix": (CB["green"], "s"),
              "encoder": (CB["purple"], "D"), "negatives": (CB["orange"], "P"),
              "oracle": (CB["blue"], "*"), "other": (CB["grey"], "^")}
    seen = set()
    for p in pts:
        col, mk = styles.get(p.get("kind", "other"), styles["other"])
        lab = p.get("kind") if p.get("kind") not in seen else None
        seen.add(p.get("kind"))
        ax.scatter(p["pm_rk"], max(p["mrr"], 1e-6), c=col, marker=mk,
                   s=110 if p.get("kind") == "oracle" else 34, label=lab,
                   edgecolors="white", linewidths=0.5, zorder=3)
    ch = _chance(res)
    if ch:
        ax.axhline(ch, ls=":", c=CB["black"], lw=0.9, zorder=1)
        ax.text(0.98, ch * 1.3, "chance", transform=ax.get_yaxis_transform(),
                fontsize=6.5, va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"pooled effective rank  $r_{\mathrm{pool}}$")
    ax.set_ylabel("MRR (Protocol R)")
    ax.set_title("Effective rank does not predict retrieval")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.62))
    _save(fig, "fig1_rank_vs_mrr")


# ------------------------------------------------------------------- fig 2
def fig_redundancy(res):
    """v3.1: (a) the severity law, (b) query collapse, (c) geometry control."""
    rho = res.get("rho", {})
    rows = rho.get("rows_trained") or rho.get("rows")
    if not rows or "MRR_trained" not in rows[0]:
        return _fig_redundancy_legacy(res)
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 2.7))
    ms = sorted({r["m"] for r in rows})
    cols = [CB["blue"], CB["orange"], CB["green"], CB["purple"]]
    for j, m in enumerate(ms):
        sub = sorted([r for r in rows if r["m"] == m], key=lambda r: r["bw_ratio"])
        x = [max(r["bw_ratio"], 1e-3) for r in sub]
        c = cols[j % len(cols)]
        axs[0].plot(x, [max(r["MRR_ridge"], 1e-6) for r in sub], "--s", c=c,
                    alpha=0.75, label=fr"ridge ceiling, $m={m}$")
        axs[0].plot(x, [max(r["MRR_trained"], 1e-6) for r in sub], "-o", c=c,
                    label=fr"trained (budgeted), $m={m}$")
        axs[1].plot(x, [r["query_eff_rank_trained"] for r in sub], "-o", c=c,
                    label=fr"trained, $m={m}$")
        axs[1].plot(x, [r["bank_eff_rank"] for r in sub], ":", c=c, alpha=0.6,
                    label=fr"candidate bank, $m={m}$")
    ctrl = rho.get("rows_geometry") or []
    for j, m in enumerate(sorted({r["m"] for r in ctrl})):
        sub = sorted([r for r in ctrl if r["m"] == m],
                     key=lambda r: r.get("dc_ratio", 0))
        axs[2].plot([max(r["dc_ratio"], 1e-3) for r in sub],
                    [max(r["MRR_raw"], 1e-6) for r in sub], "-o",
                    c=cols[j % len(cols)], label=fr"$m={m}$")
    bw = _measured_bw_ratio(res)
    if bw:
        for ax in (axs[0], axs[1]):
            ax.axvline(bw, ls="--", c=CB["red"], lw=1.0)
            ax.text(bw, 0.04, " corpus ", rotation=90, fontsize=5.8, color=CB["red"],
                    va="bottom", ha="left", transform=ax.get_xaxis_transform())
    for ax in axs:
        ax.set_xscale("log")
    axs[0].set_xlabel(r"between/within variance ratio  $\rho/(1-\rho)$")
    axs[0].set_yscale("log"); axs[0].set_ylabel("MRR (Protocol R)")
    axs[0].set_title("(a) severity law: the ceiling holds,\nthe budgeted learner collapses")
    axs[0].legend(fontsize=5.6, loc="lower left")
    axs[1].set_xlabel(r"between/within variance ratio")
    axs[1].set_yscale("log"); axs[1].set_ylabel("effective rank")
    axs[1].set_title("(b) the query bank collapses,\nthe candidate bank does not")
    axs[1].legend(fontsize=5.6)
    axs[2].set_xlabel(r"DC ratio  $\|\mu\|/\mathbb{E}\|x-\mu\|$")
    axs[2].set_yscale("log"); axs[2].set_ylabel("MRR (true-context query)")
    axs[2].set_ylim(1e-6, 2.0)
    axs[2].set_title("(c) control: bank separability is\nunaffected by $\\rho$")
    axs[2].legend(fontsize=6)
    _save(fig, "fig2_redundancy")


def _fig_redundancy_legacy(res):
    rho = res.get("rho", {})
    rows = rho.get("rows_geometry") or rho.get("rows")
    if not rows:
        return
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ms = sorted({r["m"] for r in rows})
    cols = [CB["blue"], CB["orange"], CB["green"], CB["purple"]]
    for j, m in enumerate(ms):
        sub = sorted([r for r in rows if r["m"] == m], key=lambda r: r["dc_ratio"])
        x = [max(r["dc_ratio"], 1e-3) for r in sub]
        axs[0].plot(x, [max(r["MRR_raw"], 1e-6) for r in sub], "-o",
                    c=cols[j % len(cols)], label=fr"raw, $m={m}$")
        axs[1].plot(x, [r["erank_pooled"] for r in sub], "-o", c=cols[j % len(cols)],
                    label=fr"$m={m}$")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel(r"DC ratio $\|\mu\|/\mathbb{E}\|x-\mu\|$")
    axs[0].set_yscale("log"); axs[0].set_ylabel("MRR"); axs[0].legend(fontsize=6)
    axs[1].set_ylabel("effective rank"); axs[1].legend(fontsize=6)
    _save(fig, "fig2_redundancy")


# ------------------------------------------------------------------- fig 3
def fig_ladder(res):
    rows = []
    for k, v in (res.get("protocolr") or {}).items():
        if isinstance(v, dict) and "micro" in v:
            rows.append((k.replace("_", " "), v["micro"]["MRR"], CB["red"]))
    for r in (res.get("encoder") or {}).get("rows", []):
        rows.append((f"enc:{r['tag']}", r["MRR"][0], CB["purple"]))
    for r in (res.get("fix") or {}).get("rows", []):
        rows.append((f"{r['pooling']}/{r['loss']}" + ("+vic" if r["target_vic"] else ""),
                     r["MRR"][0], CB["green"]))
    for r in (res.get("negatives") or {}).get("rows", []):
        rows.append((f"neg:{r['neg_mode']}", r["MRR"][0], CB["orange"]))
    for t, v in ((res.get("centering") or {}).get("by_transform", {})).items():
        if t != "raw":
            rows.append((f"frame: {t}", v["MRR"][0], CB["yellow"]))
    for t, v in ((res.get("latent") or {}).get("taps", {})).items():
        rows.append((f"tap: {t}", v["micro"]["MRR"], CB["sky"]))
        pm = (v.get("probe_micro") or {}).get("MRR")
        if pm is not None and np.isfinite(pm):
            rows.append((f"probe: {t}", pm, CB["grey"]))
    o = _oracle_level(res)
    if o:
        rows.append(("oracle (training-free)", o, CB["blue"]))
    if not rows:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.8, 0.22 * len(rows) + 1.0))
    y = np.arange(len(rows))
    ax.barh(y, [max(r[1], 1e-6) for r in rows], color=[r[2] for r in rows], height=0.68)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=5.8)
    ax.invert_yaxis(); ax.set_xscale("log")
    ax.set_xlabel("MRR (Protocol R, log scale)")
    ch = _chance(res)
    if ch:
        ax.axvline(ch, ls=":", c=CB["black"], lw=0.9)
    if o:
        ax.axvline(o, ls="--", c=CB["blue"], lw=0.9)
    ax.set_title("Diagnostic ladder")
    _save(fig, "fig3_ladder")


# ------------------------------------------------------------------- fig 4
def fig_spectra(res):
    sp = ART["spectra"]
    if not sp:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    cols = {"node_baseline": CB["orange"], "pooled_baseline": CB["red"],
            "pooled_fix": CB["green"], "oracle_features": CB["blue"],
            "pooled_encoder_best": CB["purple"], "pooled_neg_best": CB["sky"]}
    for k, v in sp.items():
        v = np.asarray(v)
        if v.ndim != 1 or v.size == 0:
            continue
        ax.plot(np.arange(1, v.size + 1), v, c=cols.get(k, CB["grey"]),
                label=k.replace("_", " "))
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlabel("singular-value index"); ax.set_ylabel("normalised singular value")
    ax.set_title("Node and pooled spectra coincide: pooling is not the culprit")
    ax.legend(loc="lower left", fontsize=6)
    _save(fig, "fig4_spectra")


# ------------------------------------------------------------------- fig 5
def fig_embedding_geometry(res):
    pc = {k: np.asarray(v) for k, v in ART["pca"].items() if v is not None}
    if not pc:
        return
    plt = _plt()
    keys = [k for k in ("pooled_baseline", "pooled_neg_best", "pooled_fix",
                        "oracle_features") if k in pc][:3] or list(pc)[:3]
    fig, axs = plt.subplots(1, len(keys), figsize=(2.3 * len(keys), 2.4))
    if len(keys) == 1:
        axs = [axs]
    for ax, k in zip(axs, keys):
        X = pc[k]
        ax.scatter(X[:, 0], X[:, 1], s=1.2, alpha=0.28, c=CB["blue"], linewidths=0,
                   rasterized=True)
        ax.set_title(k.replace("_", " "), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlabel("PC1", fontsize=7); ax.set_ylabel("PC2", fontsize=7)
    fig.suptitle("Trained patches form one point-mass per aspect: "
                 "the representation encodes aspect, not paper", fontsize=8)
    _save(fig, "fig5_embedding_geometry")


# ------------------------------------------------------------------- fig 6
def fig_peraspect(res):
    pa = res.get("peraspect", {})
    aspects = [a for a in pa if isinstance(pa[a], dict) and "node_rk" in pa[a]]
    if not aspects:
        return
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4))
    x = np.arange(len(aspects)); w = 0.36
    axs[0].bar(x - w / 2, [pa[a]["node_rk"][0] for a in aspects], w,
               yerr=[pa[a]["node_rk"][1] for a in aspects],
               color=CB["orange"], label="node rank", capsize=2)
    axs[0].bar(x + w / 2, [pa[a]["pool_rk"][0] for a in aspects], w,
               yerr=[pa[a]["pool_rk"][1] for a in aspects],
               color=CB["red"], label="pooled rank", capsize=2)
    if all("query_rk" in pa[a] for a in aspects):
        axs[0].plot(x, [pa[a]["query_rk"][0] for a in aspects], "kD--", ms=4,
                    label="query rank")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([f"{a}\n(m={pa[a]['members_per_patch']:.2f})" for a in aspects],
                           fontsize=7)
    axs[0].set_ylabel("effective rank"); axs[0].set_yscale("log")
    axs[0].set_title("(a) pooling is the identity for singletons"); axs[0].legend(fontsize=6)
    axs[1].bar(x, [max(pa[a]["MRR"][0], 1e-6) for a in aspects], 0.55,
               yerr=[pa[a]["MRR"][1] for a in aspects], color=CB["blue"], capsize=2)
    axs[1].set_xticks(x); axs[1].set_xticklabels(aspects, fontsize=7)
    axs[1].set_yscale("log"); axs[1].set_ylabel("MRR")
    ch = _chance(res)
    if ch:
        axs[1].axhline(ch, ls=":", c=CB["black"], lw=0.9)
    axs[1].set_title("(b) rank 47 and rank 2 fail identically")
    _save(fig, "fig6_peraspect")


# ------------------------------------------------------------------- fig 7
def fig_rank_cdf(res):
    rk = {k: np.asarray(v) for k, v in ART["ranks"].items() if np.asarray(v).size}
    if not rk:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    cols = {"baseline": CB["red"], "fix": CB["green"], "oracle": CB["blue"],
            "baseline_center": CB["orange"], "encoder_best": CB["purple"],
            "neg_best": CB["sky"], "latent_pred": CB["yellow"]}
    for k, v in rk.items():
        v = np.sort(v)
        ax.plot(v, np.arange(1, v.size + 1) / v.size, c=cols.get(k, CB["grey"]),
                label=k.replace("_", " "))
    n = res.get("config", {}).get("P")
    if n:
        xs = np.logspace(0, math.log10(n), 50)
        ax.plot(xs, xs / n, ls=":", c=CB["black"], lw=0.9, label="uniform (random)")
    ax.set_xscale("log"); ax.set_xlabel("rank of the gold patch (log)")
    ax.set_ylabel("fraction of queries $\\leq$ rank")
    ax.set_title("Trained model tracks the uniform ranker exactly")
    ax.legend(loc="upper left", fontsize=6)
    _save(fig, "fig7_rank_cdf")


# ------------------------------------------------------------------- fig 8
def fig_probe_vs_mrr(res):
    pts = [p for p in ART["scatter"] if p.get("probe") is not None]
    if not pts:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for p in pts:
        col = {"baseline": CB["red"], "fix": CB["green"], "encoder": CB["purple"],
               "negatives": CB["orange"]}.get(p.get("kind"), CB["grey"])
        ax.scatter(p["probe"], max(p["mrr"], 1e-6), c=col, s=34,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(p["name"], (p["probe"], max(p["mrr"], 1e-6)), fontsize=5.0,
                    xytext=(3, 2), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("linear-probe accuracy"); ax.set_ylabel("MRR (Protocol R)")
    ax.set_title("Probing accuracy carries no information about retrieval")
    ch = _chance(res)
    if ch:
        ax.axhline(ch, ls=":", c=CB["black"], lw=0.9)
    _save(fig, "fig8_probe_vs_mrr")


# ------------------------------------------------------------------- fig 9
def fig_oracle_controls(res):
    o = res.get("oracle", {})
    if not o:
        return
    base = o.get("whitened_with_summary") or o.get("raw_with_summary") or {}
    aspects = [a for a in base if isinstance(base[a], dict) and "MRR" in base[a]
               and a != "micro"]
    if not aspects:
        return
    series = [("oracle (ctx)", base, CB["blue"]),
              ("no-summary",
               o.get("whitened_no_summary", {}) or o.get("raw_no_summary", {}), CB["sky"]),
              ("BM25", o.get("bm25", {}), CB["orange"]),
              ("no-overlap", o.get("no_overlap", {}), CB["green"])]
    series = [s for s in series if s[1]]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(len(aspects)); w = 0.8 / max(len(series), 1)
    for j, (lab, d, col) in enumerate(series):
        vals = [d.get(a, {}).get("MRR", np.nan) for a in aspects]
        xs = x + (j - (len(series) - 1) / 2) * w
        ax.bar(xs, [max(v, 1e-6) if np.isfinite(v) else 1e-6 for v in vals], w,
               color=col, label=lab)
        for xi, v in zip(xs, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}", (xi, v), fontsize=4.6, ha="center",
                            va="bottom", rotation=90)
    ch = _chance(res)
    if ch:
        ax.axhline(ch, ls=":", c=CB["black"], lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(aspects, fontsize=7)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 3.0)
    ax.set_ylabel("MRR (Protocol R)")
    ax.set_title("Oracle leakage controls"); ax.legend(ncol=2, fontsize=6.2)
    _save(fig, "fig9_oracle_controls")


# ------------------------------------------------------------------ fig 10
def fig_fix_bars(res):
    rows = (res.get("fix") or {}).get("rows", [])
    if not rows:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.8, 0.30 * len(rows) + 1.0))
    names = [f"{r['pooling']}/{r['loss']}" + ("+tgt-vic" if r["target_vic"] else "")
             for r in rows]
    y = np.arange(len(rows)); h = 0.38
    ax.barh(y - h / 2, [max(r["MRR"][0], 1e-6) for r in rows], h,
            xerr=[r["MRR"][1] for r in rows], color=CB["green"], capsize=2,
            label="raw cosine")
    ax.barh(y + h / 2, [max(r.get("MRR_center", [1e-6])[0], 1e-6) for r in rows], h,
            xerr=[r.get("MRR_center", [0, 0])[1] for r in rows], color=CB["orange"],
            capsize=2, label="centred cosine")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.8); ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("MRR (Protocol R), mean $\\pm$ std over seeds")
    o = _oracle_level(res)
    if o:
        ax.axvline(o, ls="--", c=CB["blue"], lw=0.9)
    ch = _chance(res)
    if ch:
        ax.axvline(ch, ls=":", c=CB["black"], lw=0.9)
    ax.set_title("Seven interventions, one flat line")
    ax.legend(loc="lower right", fontsize=6.2)
    _save(fig, "fig10_fix_bars")


# ------------------------------------------------------------------ fig 11
def fig_training_dynamics(res):
    hs = {k: np.asarray(v) for k, v in ART["history"].items() if np.asarray(v).size}
    if not hs:
        return
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4))
    cols = [CB["red"], CB["green"], CB["orange"], CB["purple"], CB["sky"]]
    for j, (k, H) in enumerate(hs.items()):
        if H.ndim != 2 or H.shape[1] < 4:
            continue
        c = cols[j % len(cols)]
        axs[0].plot(H[:, 0], H[:, 1], c=c, label=k.replace("_", " "))
        axs[1].plot(H[:, 0], H[:, 2], c=c, label=f"{k} pooled")
        axs[1].plot(H[:, 0], H[:, 3], c=c, ls="--", alpha=0.6, label=f"{k} node")
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("training loss")
    axs[0].set_title("(a) the objective is satisfied"); axs[0].legend(fontsize=6)
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("effective rank")
    axs[1].set_yscale("log")
    axs[1].set_title("(b) low rank is present at epoch 0, not learned")
    axs[1].legend(fontsize=5.6)
    _save(fig, "fig11_training_dynamics")


# ------------------------------------------------------------------ fig 12
def fig_encoder_depth(res):
    rows = (res.get("encoder") or {}).get("rows", [])
    if not rows:
        return
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    tags = [r["tag"] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    axs[0].bar(x - w / 2, [max(r["MRR"][0], 1e-6) for r in rows], w,
               yerr=[r["MRR"][1] for r in rows], color=CB["purple"], capsize=2,
               label="raw cosine")
    axs[0].bar(x + w / 2, [max(r.get("MRR_center", [1e-6])[0], 1e-6) for r in rows], w,
               yerr=[r.get("MRR_center", [0, 0])[1] for r in rows], color=CB["orange"],
               capsize=2, label="centred")
    ch = _chance(res)
    if ch:
        axs[0].axhline(ch, ls=":", c=CB["black"], lw=0.9)
    o = _oracle_level(res)
    if o:
        axs[0].axhline(o, ls="--", c=CB["blue"], lw=0.9)
    axs[0].set_xticks(x); axs[0].set_xticklabels(tags, rotation=30, ha="right", fontsize=6.2)
    axs[0].set_yscale("log"); axs[0].set_ylabel("MRR (Protocol R)")
    axs[0].set_title("(a) retrieval vs encoder depth"); axs[0].legend(fontsize=6)
    axs[1].bar(x - 0.18, [r["node_rk"][0] for r in rows], 0.34, color=CB["orange"],
               label="node rank")
    axs[1].bar(x + 0.18, [r["pm_rk"][0] for r in rows], 0.34, color=CB["red"],
               label="pooled rank")
    axs[1].set_xticks(x); axs[1].set_xticklabels(tags, rotation=30, ha="right", fontsize=6.2)
    axs[1].set_yscale("log"); axs[1].set_ylabel("effective rank")
    axs[1].set_title("(b) message passing destroys rank"); axs[1].legend(fontsize=6)
    _save(fig, "fig12_encoder_depth")


# ------------------------------------------------------------------ fig 13
def fig_centering(res):
    ce = res.get("centering") or {}
    bt = ce.get("by_transform")
    if not bt:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ts = list(bt); x = np.arange(len(ts))
    ax.bar(x, [max(bt[t]["MRR"][0], 1e-6) for t in ts],
           yerr=[bt[t]["MRR"][1] for t in ts], color=CB["orange"], capsize=2, width=0.6)
    ch = _chance(res)
    if ch:
        ax.axhline(ch, ls=":", c=CB["black"], lw=0.9)
    o = _oracle_level(res)
    if o:
        ax.axhline(o, ls="--", c=CB["blue"], lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(ts, rotation=20, ha="right", fontsize=7)
    ax.set_yscale("log"); ax.set_ylabel("MRR (Protocol R)")
    st = ce.get("bank_stats_mean", {})
    sub = (f"cand DC {_fmt(st.get('cand_dc_ratio'), 1)}, "
           f"query DC {_fmt(st.get('query_dc_ratio'), 1)}, "
           f"query erank {_fmt(st.get('query_eff_rank'), 1)}") if st else ""
    ax.set_title("A constant query cannot be rescued by any frame\n" + sub, fontsize=7.2)
    _save(fig, "fig13_centering")


# ------------------------------------------------------------------ fig 14
def fig_rank_vs_degree(res):
    ri = res.get("rankinit") or {}
    per, deg = ri.get("per_depth"), ri.get("degrees")
    if not per or not deg:
        return
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    depths = sorted(per, key=lambda k: str(k))
    cols = [CB["blue"], CB["orange"], CB["green"], CB["red"], CB["purple"]]
    ntypes = sorted({nt for d in per.values() for nt in d})
    for j, d in enumerate(depths):
        xs, ys = [], []
        for nt in ntypes:
            if nt not in per[d] or nt not in deg:
                continue
            xs.append(max(deg[nt]["mean_in_degree"], 1e-2))
            ys.append(per[d][nt]["eff_rank"])
        order = np.argsort(xs)
        axs[0].plot(np.array(xs)[order], np.array(ys)[order], "-o",
                    c=cols[j % len(cols)], label=str(d))
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("mean in-degree of the node type")
    axs[0].set_ylabel("effective rank (untrained)")
    axs[0].set_title("(a) rank falls with degree, before training")
    axs[0].legend(fontsize=6, title="encoder", title_fontsize=6)
    for j, d in enumerate(depths):
        xs = [nt for nt in ntypes if nt in per[d]]
        axs[1].plot(range(len(xs)), [per[d][nt]["dc_ratio"] for nt in xs], "-o",
                    c=cols[j % len(cols)], label=str(d))
    axs[1].set_xticks(range(len(ntypes)))
    axs[1].set_xticklabels(ntypes, rotation=30, ha="right", fontsize=6.2)
    axs[1].set_yscale("log"); axs[1].set_ylabel(r"DC ratio $\|\mu\|/\mathbb{E}\|x-\mu\|$")
    axs[1].axhline(1.0, ls=":", c=CB["black"], lw=0.9)
    axs[1].set_title("(b) message passing inflates the shared mean")
    axs[1].legend(fontsize=6)
    _save(fig, "fig14_rank_vs_degree")


# ------------------------------------------------------------------ fig 15
def fig_dc_ratio(res):
    stages = []
    meas = (res.get("rho") or {}).get("measured", {})
    for k in ("input_raw", "input_whitened", "trained_pooled", "trained_query"):
        v = meas.get(k)
        if v and v.get("dc_ratio"):
            stages.append((k, v["dc_ratio"]))
    for t, v in ((res.get("latent") or {}).get("taps", {})).items():
        g = v.get("geometry", {})
        if g.get("dc_ratio"):
            stages.append((f"tap:{t}", g["dc_ratio"]))
    base = (res.get("oracle") or {}).get("whitened_with_summary") or {}
    for a, v in base.items():
        if isinstance(v, dict) and isinstance(v.get("dc"), dict):
            stages.append((f"oracle:{a}", v["dc"]["dc_ratio"]))
    if not stages:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.8, 0.20 * len(stages) + 1.0))
    y = np.arange(len(stages))
    ax.barh(y, [s[1] for s in stages],
            color=[CB["red"] if ("trained" in s[0] or "tap" in s[0]) else CB["blue"]
                   for s in stages], height=0.62)
    ax.axvline(1.0, ls=":", c=CB["black"], lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in stages], fontsize=6.2)
    ax.invert_yaxis(); ax.set_xscale("log")
    ax.set_xlabel(r"DC ratio $\|\mu\|/\mathbb{E}\|x-\mu\|$  (log)")
    ax.set_title("Shared-mean dominance by pipeline stage")
    _save(fig, "fig15_dc_ratio")


# ------------------------------------------------------------------ fig 16
def fig_information_ladder(res):
    lat = (res.get("latent") or {}).get("taps")
    if not lat:
        return
    order = [t for t in LATENT_TAPS if t in lat]
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.6))
    x = np.arange(len(order)); w = 0.38
    direct = [max(lat[t]["micro"]["MRR"], 1e-6) for t in order]
    probe = [max((lat[t].get("probe_micro") or {}).get("MRR", np.nan), 1e-6)
             for t in order]
    axs[0].bar(x - w / 2, direct, w, color=CB["red"], label="scored in learned space")
    axs[0].bar(x + w / 2, probe, w, color=CB["green"], label="ridge probe -> oracle")
    ch = _chance(res)
    if ch:
        axs[0].axhline(ch, ls=":", c=CB["black"], lw=0.9)
    o = _oracle_level(res)
    if o:
        axs[0].axhline(o, ls="--", c=CB["blue"], lw=0.9)
    axs[0].set_xticks(x); axs[0].set_xticklabels(order, rotation=30, ha="right", fontsize=6.2)
    axs[0].set_yscale("log"); axs[0].set_ylabel("MRR (Protocol R)")
    axs[0].set_title("(a) where paper identity dies")
    axs[0].legend(fontsize=5.8, loc="upper right")

    axs[1].bar(x, [lat[t]["geometry"].get("eff_rank", np.nan) for t in order],
               0.55, color=CB["purple"])
    axs[1].set_xticks(x); axs[1].set_xticklabels(order, rotation=30, ha="right", fontsize=6.2)
    axs[1].set_yscale("log"); axs[1].set_ylabel("query effective rank")
    axs[1].set_title("(b) the query bank collapses to a point")

    axs[2].bar(x, [lat[t]["geometry"].get("self_sim", np.nan) for t in order],
               0.55, color=CB["orange"])
    axs[2].axhline(1.0, ls=":", c=CB["black"], lw=0.9)
    axs[2].set_xticks(x); axs[2].set_xticklabels(order, rotation=30, ha="right", fontsize=6.2)
    axs[2].set_ylabel("mean pairwise cosine between queries")
    axs[2].set_title("(c) all queries are the same query")
    _save(fig, "fig16_information_ladder")


# ------------------------------------------------------------------ fig 17
def fig_negatives(res):
    rows = (res.get("negatives") or {}).get("rows", [])
    if not rows:
        return
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
    tags = [r["neg_mode"] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    axs[0].bar(x - w / 2, [max(r["MRR"][0], 1e-6) for r in rows], w,
               yerr=[r["MRR"][1] for r in rows], color=CB["orange"], capsize=2,
               label="raw cosine")
    axs[0].bar(x + w / 2, [max(r.get("MRR_center", [1e-6])[0], 1e-6) for r in rows], w,
               yerr=[r.get("MRR_center", [0, 0])[1] for r in rows],
               color=CB["sky"], capsize=2, label="centred")
    ch = _chance(res)
    if ch:
        axs[0].axhline(ch, ls=":", c=CB["black"], lw=0.9)
    o = _oracle_level(res)
    if o:
        axs[0].axhline(o, ls="--", c=CB["blue"], lw=0.9)
    axs[0].set_xticks(x); axs[0].set_xticklabels(tags, rotation=30, ha="right", fontsize=6.2)
    axs[0].set_yscale("log"); axs[0].set_ylabel("MRR (Protocol R)")
    axs[0].set_title("(a) does removing the aspect shortcut help?")
    axs[0].legend(fontsize=6)
    axs[1].bar(x - w / 2, [r["pm_rk"][0] for r in rows], w, color=CB["red"],
               label="pooled rank")
    axs[1].bar(x + w / 2, [r["query_rk"][0] for r in rows], w, color=CB["purple"],
               label="query rank")
    axs[1].set_xticks(x); axs[1].set_xticklabels(tags, rotation=30, ha="right", fontsize=6.2)
    axs[1].set_yscale("log"); axs[1].set_ylabel("effective rank")
    axs[1].set_title("(b) query-side collapse vs negatives")
    axs[1].legend(fontsize=6)
    _save(fig, "fig17_negatives")


# ------------------------------------------------------------------ fig 18
def fig_poolsize(res):
    ps = (res.get("poolsize") or {}).get("curves")
    if not ps:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    cols = {"baseline": CB["red"], "fix": CB["green"], "oracle": CB["blue"],
            "encoder_best": CB["purple"], "neg_best": CB["orange"],
            "baseline_center": CB["yellow"], "latent_pred": CB["sky"]}
    for k, rows in ps.items():
        N = [r["N"] for r in rows]; M = [max(r["MRR"], 1e-7) for r in rows]
        ax.plot(N, M, "-o", c=cols.get(k, CB["grey"]), label=k.replace("_", " "), ms=3)
    Ns = sorted({r["N"] for rows in ps.values() for r in rows})
    if Ns:
        ax.plot(Ns, [chance_mrr(n) for n in Ns], ls=":", c=CB["black"], lw=1.0,
                label="chance")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("candidate-pool size $N$"); ax.set_ylabel("MRR")
    ax.set_title("How many items can the model actually separate?")
    ax.legend(fontsize=5.8, loc="lower left")
    _save(fig, "fig18_poolsize")


# ------------------------------------------------------------------ fig 19
def fig_variance(res):
    src = {}
    pv = ((res.get("protocolr") or {}).get("euclidean_mean_raw") or {}).get("pooled_variance")
    if pv:
        src["trained pooled (raw feats)"] = pv
    pw = ((res.get("protocolr") or {}).get("euclidean_mean_white") or {}).get("pooled_variance")
    if pw:
        src["trained pooled (whitened)"] = pw
    for k, v in ((res.get("latent") or {}).get("variance") or {}).items():
        src[k] = v
    if not src:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.8, 0.34 * len(src) + 1.1))
    keys = list(src)
    y = np.arange(len(keys))
    betw = [src[k]["between_frac"] for k in keys]
    wit = [src[k]["within_frac"] for k in keys]
    ax.barh(y, betw, color=CB["red"], height=0.6, label="between aspect (shortcut)")
    ax.barh(y, wit, left=betw, color=CB["green"], height=0.6,
            label="between paper (what retrieval needs)")
    for yi, (b, w) in enumerate(zip(betw, wit)):
        ax.annotate(f"{100*w:.2f}%", (min(b + w, 1.0), yi), fontsize=6,
                    va="center", ha="right", color="white")
    ax.set_yticks(y); ax.set_yticklabels(keys, fontsize=6.5); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("fraction of representation variance")
    ax.set_title("The objective optimises the wrong variance component")
    ax.legend(fontsize=6, loc="lower right")
    _save(fig, "fig19_variance")


ALL_FIGS = [fig_rank_vs_mrr, fig_redundancy, fig_ladder, fig_spectra,
            fig_embedding_geometry, fig_peraspect, fig_rank_cdf, fig_probe_vs_mrr,
            fig_oracle_controls, fig_fix_bars, fig_training_dynamics,
            fig_encoder_depth, fig_centering, fig_rank_vs_degree, fig_dc_ratio,
            fig_information_ladder, fig_negatives, fig_poolsize, fig_variance]


def make_all_figures(res):
    print("\n" + "#" * 100 + "\n#  FIGURES\n" + "#" * 100)
    os.makedirs(FIG_DIR, exist_ok=True)
    for fn in ALL_FIGS:
        try:
            fn(res)
        except Exception as e:                              # noqa: BLE001
            print(f"  [warn] {fn.__name__} skipped: {e}")
    print(f"  all figures -> {FIG_DIR}")


def save_artifacts(path):
    flat = {}
    for grp in ("spectra", "pca", "ranks", "history"):
        for k, v in ART[grp].items():
            if v is None:
                continue
            try:                                            # FIX(v3.1): tolerate ragged
                arr = np.asarray(v, dtype=np.float32)
                if arr.dtype == object:
                    raise TypeError("object array")
                flat[f"{grp}__{k}"] = arr
            except Exception as e:                          # noqa: BLE001
                print(f"  [warn] artifact {grp}/{k} not serialisable: {e}")
    flat["scatter__json"] = np.array([json.dumps(ART["scatter"])])
    np.savez_compressed(path, **flat)
    print(f"[save] {path}")


def load_artifacts(path):
    if not os.path.exists(path):
        print(f"  [warn] no artifacts at {path}"); return
    z = np.load(path, allow_pickle=False)
    for key in z.files:
        if key == "scatter__json":
            ART["scatter"] = json.loads(str(z[key][0])); continue
        grp, name = key.split("__", 1)
        ART.setdefault(grp, {})[name] = z[key]
    print(f"[load] artifacts from {path}")


# ============================================================================
# 12. STAGES
# ============================================================================
def stage_rankinit(ctx, args):
    deg = node_type_degrees(ctx["data"])
    print("  in-degree per node type:")
    for nt, v in sorted(deg.items(), key=lambda kv: kv[1]["mean_in_degree"]):
        print(f"    {nt:<12s} mean_in_degree={v['mean_in_degree']:7.2f} n={v['n_nodes']}")
    per = {}
    for tag, depth, resid, mlp_in in [("input(d0,linear)", 0, False, False),
                                      ("d1", 1, False, True), ("d2", 2, False, True),
                                      ("d3", 3, False, True), ("d2+res", 2, True, True)]:
        try:
            per[tag] = untrained_encoder_ranks(ctx, depth, residual=resid,
                                               mlp_in=mlp_in, seed=0)
            print(f"    [{tag:<16s}] erank  " +
                  "  ".join(f"{nt}:{per[tag][nt]['eff_rank']:.1f}"
                            for nt in sorted(per[tag])))
            print(f"    {'':18s}  dc     " +
                  "  ".join(f"{nt}:{per[tag][nt]['dc_ratio']:.1f}"
                            for nt in sorted(per[tag])))
        except Exception as e:                              # noqa: BLE001
            print(f"    [warn] {tag} failed: {e}")
    inp = {}
    for nt, x in (ctx.get("raw_x") or {}).items():
        xs = _subsample(x, SPEC_ROWS)
        inp[nt] = {"eff_rank": float(eff_rank_only(xs)), **dc_stats(xs)}
    return {"degrees": deg, "per_depth": per, "input_features": inp}


def stage_protocolr(ctx, args):
    out = {}
    for ci, (name, whiten) in enumerate([("euclidean_mean_raw", False),
                                         ("euclidean_mean_white", True)]):
        MRR, MRRC, PMRK, ACC, per, dcs, var = [], [], [], [], [], [], []
        for si, s in enumerate(args.seeds):
            track = (ci == 0 and si == 0)
            model, pooler, pr, pe, pres_d, mask_idx, diag, tm, hist = train_variant(
                s, ctx["data"], ctx["active"], ctx["membership"], ctx["pres"], ctx["rwse"],
                ctx["maskable"], ctx["pe_dim"], pool_mode="mean", loss_mode="cos",
                whitened_x=ctx["whitened_x"] if whiten else None,
                epochs=args.epochs, track=track)
            r = protocol_r_model(model, pr, pe, pres_d, ctx["active"], ctx["queries"], seed=s)
            p = eval_probe(pr, pres_d, ctx["active"], ctx["label"], s)
            MRR.append(r["micro_by_transform"]["raw"]["MRR"])
            MRRC.append(r["micro_by_transform"]["center"]["MRR"])
            PMRK.append(diag["patchmean_eff_rank"]); ACC.append(p["acc"])
            per.append(r["per_aspect_by_transform"]["raw"]); dcs.append(diag["pooled_dc"])
            var.append(diag["pooled_variance"])
            print(f"  [{name}] seed {s} | probe {p['acc']:.3f} "
                  f"| MRR raw {MRR[-1]:.5f} centred {MRRC[-1]:.5f} "
                  f"| chance {r['micro']['chance_MRR']:.2e} "
                  f"| pm_rk {diag['patchmean_eff_rank']:.2f} "
                  f"| within-paper var {100*diag['pooled_variance']['within_frac']:.2f}% "
                  f"| p={r['micro']['p_value_vs_null']:.3f} | {tm['train_sec']:.0f}s",
                  flush=True)
            if track:
                ART["history"]["baseline"] = np.asarray(hist, dtype=np.float32)
                sp, _ = spectrum_of(diag["pooled_flat_ref"])
                if sp is not None:
                    ART["spectra"]["pooled_baseline"] = sp
                nz = torch.cat([diag["node_latents_ref"][a] for a in ctx["active"]], 0)
                spn, _ = spectrum_of(nz)
                if spn is not None:
                    ART["spectra"]["node_baseline"] = spn
                ART["pca"]["pooled_baseline"] = pca2_of(diag["pooled_flat_ref"])
                ART["ranks"]["baseline"] = r["ranks_by_transform"]["raw"][:RANK_KEEP]
                ART["ranks"]["baseline_center"] = \
                    r["ranks_by_transform"]["center"][:RANK_KEEP]
            del model, pooler, pr, diag
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        out[name] = {
            "micro": {"MRR": float(np.mean(MRR)), "MRR_std": float(np.std(MRR)),
                      "chance_MRR": chance_mrr(ctx["P"]),
                      "p_value_vs_null": float(np.max(
                          [max(v[a]["p_value_vs_null"] for a in v) for v in per]))},
            "MRR_center": _mm(MRRC), "probe_acc": _mm(ACC), "pm_rk": _mm(PMRK),
            "pooled_dc": {k: float(np.mean([d[k] for d in dcs])) for k in dcs[0]},
            "pooled_variance": {k: float(np.mean([v[k] for v in var])) for k in var[0]},
            "per_aspect_last_seed": per[-1]}
        ART["scatter"].append({"name": name, "pm_rk": float(np.mean(PMRK)),
                               "mrr": float(np.mean(MRR)), "probe": float(np.mean(ACC)),
                               "kind": "baseline"})
    return out


def stage_peraspect(ctx, args):
    acc = defaultdict(lambda: defaultdict(list))
    for s in args.seeds:
        model, pooler, pr, pe, pres_d, mask_idx, diag, tm, _ = train_variant(
            s, ctx["data"], ctx["active"], ctx["membership"], ctx["pres"], ctx["rwse"],
            ctx["maskable"], ctx["pe_dim"], pool_mode="mean", loss_mode="cos",
            epochs=args.epochs)
        r = protocol_r_model(model, pr, pe, pres_d, ctx["active"], ctx["queries"], seed=s)
        for a in ctx["active"]:
            acc[a]["node_rk"].append(diag["node_eff_rank_per_aspect"][a])
            acc[a]["pool_rk"].append(diag["pool_eff_rank_per_aspect"][a])
            acc[a]["MRR"].append(
                r["per_aspect_by_transform"]["raw"].get(a, {}).get("MRR", np.nan))
            acc[a]["MRR_center"].append(
                r["per_aspect_by_transform"]["center"].get(a, {}).get("MRR", np.nan))
            bs = r["bank_stats"].get(a, {})
            acc[a]["node_dc"].append(diag["node_dc_per_aspect"][a]["dc_ratio"])
            acc[a]["query_dc"].append(bs.get("query_dc", {}).get("dc_ratio", np.nan))
            acc[a]["query_rk"].append(bs.get("query_eff_rank", np.nan))
            acc[a]["margin_z"].append(bs.get("gold_margin_z", np.nan))
        print(f"  [peraspect] seed {s} done ({tm['train_sec']:.0f}s)", flush=True)
        del model, pooler, pr, diag
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = {}
    for a in ctx["active"]:
        row = {k: [float(np.nanmean(v)), float(np.nanstd(v))] for k, v in acc[a].items()}
        row["members_per_patch"] = float(ctx["counts"][a].float().mean())
        out[a] = row
        print(f"    {a:<7s} m={row['members_per_patch']:5.2f} "
              f"node_rk={row['node_rk'][0]:6.2f} pool_rk={row['pool_rk'][0]:6.2f} "
              f"query_rk={row['query_rk'][0]:6.2f} "
              f"MRR={row['MRR'][0]:.5f} centred={row['MRR_center'][0]:.5f} "
              f"| margin_z={row['margin_z'][0]:+.3f}")
    return out


def stage_centering(ctx, args):
    seeds = args.seeds[:max(1, args.centering_max_seeds)]
    per_t, bank = defaultdict(list), defaultdict(list)
    for s in seeds:
        model, pooler, pr, pe, pres_d, mask_idx, diag, tm, _ = train_variant(
            s, ctx["data"], ctx["active"], ctx["membership"], ctx["pres"], ctx["rwse"],
            ctx["maskable"], ctx["pe_dim"], pool_mode="mean", loss_mode="cos",
            epochs=args.epochs)
        r = protocol_r_model(model, pr, pe, pres_d, ctx["active"], ctx["queries"],
                             seed=s, transforms=CENTERING_TRANSFORMS)
        for t in CENTERING_TRANSFORMS:
            per_t[t].append(r["micro_by_transform"][t]["MRR"])
        for a, st in r["bank_stats"].items():
            bank["cand_dc_ratio"].append(st["cand_dc"]["dc_ratio"])
            bank["cand_dc_energy"].append(st["cand_dc"]["dc_energy"])
            bank["query_dc_ratio"].append(st["query_dc"]["dc_ratio"])
            bank["query_dc_energy"].append(st["query_dc"]["dc_energy"])
            bank["cand_eff_rank"].append(st["cand_eff_rank"])
            bank["query_eff_rank"].append(st["query_eff_rank"])
            bank["query_self_similarity"].append(st["query_self_similarity"])
        print("  [centering] seed %d | %s | %.0fs" % (
            s, " ".join(f"{t}={r['micro_by_transform'][t]['MRR']:.5f}"
                        for t in CENTERING_TRANSFORMS), tm["train_sec"]), flush=True)
        del model, pooler, pr, diag
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    by_t = {t: {"MRR": _mm(v)} for t, v in per_t.items()}
    st = {k: float(np.nanmean(v)) for k, v in bank.items()}
    print("\n  ---- RETRIEVAL-FRAME SWEEP (Protocol R) ----")
    for t in CENTERING_TRANSFORMS:
        print(f"    {t:<10s} MRR {by_t[t]['MRR'][0]:.5f} ± {by_t[t]['MRR'][1]:.5f}")
    print(f"    candidate bank: DC {st.get('cand_dc_ratio', float('nan')):.2f} "
          f"erank {st.get('cand_eff_rank', float('nan')):.1f}")
    print(f"    query bank    : DC {st.get('query_dc_ratio', float('nan')):.2f} "
          f"erank {st.get('query_eff_rank', float('nan')):.1f} "
          f"self-sim {st.get('query_self_similarity', float('nan')):.3f}")
    best = max(by_t, key=lambda t: by_t[t]["MRR"][0])
    print(f"    >>> best frame: {best} "
          f"({by_t[best]['MRR'][0] / max(by_t['raw']['MRR'][0], 1e-12):.1f}x over raw)")
    if st.get("query_self_similarity", 0) > 0.9:
        print("    NOTE: the query bank is essentially ONE vector; no frame can help, "
              "because a constant query induces the same ranking for every query.")
    return {"by_transform": by_t, "bank_stats_mean": st, "best_transform": best,
            "seeds": seeds}


def stage_encoder(ctx, args):
    seeds = args.seeds[:max(1, args.enc_max_seeds)]
    rows, best, best_art = [], None, None
    for tag, depth, resid, mlp_in in ENC_VARIANTS:
        MRR, MRRC, NRK, PMRK, ACC, secs = [], [], [], [], [], []
        cand_art = None
        for si, s in enumerate(seeds):
            try:
                head, pr, pe, pres_d, mask_idx, diag, tm, hist = train_encoder_variant(
                    s, ctx, depth=depth, residual=resid, mlp_in=mlp_in,
                    pool_mode="mean", loss_mode=args.enc_loss,
                    whitened_x=ctx["whitened_x"] if args.enc_whiten else None,
                    epochs=args.epochs, track=(si == 0))
            except Exception as e:                          # noqa: BLE001
                print(f"  [ERROR] encoder {tag} seed {s}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            r = protocol_r_model(head, pr, pe, pres_d, ctx["active"], ctx["queries"], seed=s)
            p = eval_probe(pr, pres_d, ctx["active"], ctx["label"], s)
            MRR.append(r["micro_by_transform"]["raw"]["MRR"])
            MRRC.append(r["micro_by_transform"]["center"]["MRR"])
            NRK.append(diag["node_eff_rank_mean"]); PMRK.append(diag["patchmean_eff_rank"])
            ACC.append(p["acc"]); secs.append(tm["train_sec"])
            print(f"  [enc {tag:<10s}] seed {s} | probe {p['acc']:.3f} "
                  f"| MRR raw {MRR[-1]:.5f} centred {MRRC[-1]:.5f} "
                  f"| node_rk {diag['node_eff_rank_mean']:6.2f} "
                  f"| pm_rk {diag['patchmean_eff_rank']:6.2f} | {tm['train_sec']:.0f}s",
                  flush=True)
            if si == 0:
                cand_art = {"spec": spectrum_of(diag["pooled_flat_ref"])[0],
                            "ranks": r["ranks_by_transform"]["raw"][:RANK_KEEP],
                            "hist": np.asarray(hist, dtype=np.float32)}
            del head, pr, diag
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not MRR:
            continue
        row = {"tag": tag, "depth": depth, "residual": resid, "mlp_in": mlp_in,
               "MRR": _mm(MRR), "MRR_center": _mm(MRRC), "node_rk": _mm(NRK),
               "pm_rk": _mm(PMRK), "probe_acc": _mm(ACC),
               "train_sec_mean": float(np.mean(secs))}
        rows.append(row)
        ART["scatter"].append({"name": f"enc:{tag}", "pm_rk": row["pm_rk"][0],
                               "mrr": row["MRR"][0], "probe": row["probe_acc"][0],
                               "kind": "encoder"})
        if best is None or row["MRR"][0] > best["MRR"][0]:
            best, best_art = row, cand_art
    if best_art:
        if best_art.get("spec") is not None:
            ART["spectra"]["pooled_encoder_best"] = best_art["spec"]
        ART["ranks"]["encoder_best"] = best_art["ranks"]
        if best_art["hist"].size:
            ART["history"]["encoder_best"] = best_art["hist"]
    print("\n  ---- ENCODER SWEEP (Protocol R) ----")
    for r in rows:
        print(f"    {r['tag']:<12s} depth={r['depth']} res={int(r['residual'])} "
              f"| node_rk {r['node_rk'][0]:6.2f} | pm_rk {r['pm_rk'][0]:6.2f} "
              f"| MRR {r['MRR'][0]:.5f} (centred {r['MRR_center'][0]:.5f})")
    if best:
        print(f"    >>> best: {best['tag']} MRR={best['MRR'][0]:.5f}")
    return {"rows": rows, "best": best, "seeds": seeds, "loss": args.enc_loss}


def stage_latent(ctx, args):
    """
    EXPERIMENT A (localisation). Where along the pipeline does paper identity
    disappear? Direct Protocol R at each tap + a ridge probe onto the oracle
    target space (upper bound on any linear read-out).
    """
    seeds = args.seeds[:max(1, args.latent_max_seeds)]
    oracle_asp = aspect_feature_matrix(ctx["data"], ctx["active"], ctx["membership"],
                                       feats=ctx["whitened_x"], raw_x=ctx.get("raw_x"))
    acc = {t: defaultdict(list) for t in LATENT_TAPS}
    first_ranks = None
    for s in seeds:
        model, pooler, pr, pe, pres_d, mask_idx, diag, tm, _ = train_variant(
            s, ctx["data"], ctx["active"], ctx["membership"], ctx["pres"], ctx["rwse"],
            ctx["maskable"], ctx["pe_dim"], pool_mode="mean", loss_mode="cos",
            epochs=args.epochs)
        lad = information_ladder(model, pr, pe, pres_d, ctx["active"], ctx["queries"],
                                 oracle_asp, seed=s)
        for t in LATENT_TAPS:
            acc[t]["MRR"].append(lad[t]["micro"]["MRR"])
            acc[t]["probe_MRR"].append((lad[t]["probe_micro"] or {}).get("MRR", np.nan))
            for gk, gv in lad[t]["geometry"].items():
                acc[t][f"geo_{gk}"].append(gv)
        if first_ranks is None:
            first_ranks = lad["pred"]["_ranks"][:RANK_KEEP]
        print(f"  [latent] seed {s} | " +
              " ".join(f"{t}:{lad[t]['micro']['MRR']:.5f}"
                       f"/p{(lad[t]['probe_micro'] or {}).get('MRR', float('nan')):.3f}"
                       for t in LATENT_TAPS) + f" | {tm['train_sec']:.0f}s", flush=True)
        del model, pooler, pr, diag, lad
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    taps = {}
    for t in LATENT_TAPS:
        geo = {k[4:]: float(np.nanmean(v)) for k, v in acc[t].items()
               if k.startswith("geo_")}
        taps[t] = {"micro": {"MRR": float(np.nanmean(acc[t]["MRR"])),
                             "MRR_std": float(np.nanstd(acc[t]["MRR"]))},
                   "probe_micro": {"MRR": float(np.nanmean(acc[t]["probe_MRR"])),
                                   "MRR_std": float(np.nanstd(acc[t]["probe_MRR"]))},
                   "geometry": geo}
    if first_ranks is not None:
        ART["ranks"]["latent_pred"] = first_ranks

    var = {}
    try:
        P, A, D = oracle_asp.shape
        n = min(SPEC_ROWS, P)
        flat = oracle_asp[:n].reshape(-1, D)
        gid = torch.arange(A, device=oracle_asp.device).repeat(n)
        var["oracle target space"] = variance_decomposition(flat, gid)
    except Exception as e:                                  # noqa: BLE001
        print(f"  [warn] oracle variance decomposition: {e}")

    print("\n  ---- INFORMATION LADDER ----")
    for t in LATENT_TAPS:
        g = taps[t]["geometry"]
        print(f"    {t:<12s} MRR {taps[t]['micro']['MRR']:.5f} "
              f"| ridge->oracle {taps[t]['probe_micro']['MRR']:.5f} "
              f"| q_erank {g.get('eff_rank', float('nan')):6.2f} "
              f"| self-sim {g.get('self_sim', float('nan')):.3f} "
              f"| dc {g.get('dc_ratio', float('nan')):7.2f}")
    pe_only = taps["pe_only"]["micro"]["MRR"]
    pred = taps["pred"]["micro"]["MRR"]
    if np.isfinite(pred) and np.isfinite(pe_only) and \
            abs(pred - pe_only) < 0.25 * max(pred, 1e-12):
        print("    >>> VERDICT: pred ~= pe_only  ->  the query is a function of the "
              "ASPECT CODE ALONE. Paper identity never reaches the query side.")
    # FIX(v3.1): nan-safe argmax
    probe_vals = {t: (taps[t]["probe_micro"]["MRR"]
                      if np.isfinite(taps[t]["probe_micro"]["MRR"]) else -np.inf)
                  for t in LATENT_TAPS}
    best_probe = max(probe_vals, key=probe_vals.get)
    print(f"    >>> best linear read-out: {best_probe} "
          f"-> {taps[best_probe]['probe_micro']['MRR']:.4f} "
          f"(oracle {_fmt(_oracle_level(args.res_ref))})")
    del oracle_asp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"taps": taps, "variance": var, "seeds": seeds,
            "best_probe_tap": best_probe}


def stage_negatives(ctx, args):
    """
    EXPERIMENT B (the decisive fix test). Does removing the aspect shortcut from
    the negatives -- and using real full-pool negatives -- restore instance
    identity?

      cos             : cosine regression (the baseline; no negatives at all)
      inbatch_random  : InfoNCE, aspect-MIXED in-batch negatives  (the default
                        everyone uses -- a 3-way classifier solves it)
      aspect_matched  : InfoNCE, every negative is the SAME aspect, different
                        paper -> the shortcut is removed by construction
      aspect_hard     : aspect_matched + top-k hardest negatives mined from a
                        uniform sample of `neg_pool` candidates
      fullpool        : aspect_matched + `neg_pool` negatives drawn from the
                        REAL 58k candidate bank via the EMA target encoder

    If aspect_matched/hard/fullpool recover retrieval, the paper ships a
    prescription. If they do not, the null survives the strongest available
    objective and the closing control is airtight either way.
    """
    seeds = args.seeds[:max(1, args.neg_max_seeds)]
    rows, best, best_art = [], None, None

    for nm in NEG_MODES:
        MRR, MRRC, PMRK, QRK, ACC, SELF, MARG, PVAL, secs = \
            [], [], [], [], [], [], [], [], []
        cand_art = None
        for si, s in enumerate(seeds):
            try:
                model, pooler, pr, pe, pres_d, mask_idx, diag, tm, hist = train_variant(
                    s, ctx["data"], ctx["active"], ctx["membership"], ctx["pres"],
                    ctx["rwse"], ctx["maskable"], ctx["pe_dim"], pool_mode="mean",
                    loss_mode="cos", epochs=args.epochs, neg_mode=nm,
                    neg_pool=args.neg_pool, track=(si == 0))
            except Exception as e:                          # noqa: BLE001
                print(f"  [ERROR] negatives {nm} seed {s}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            r = protocol_r_model(model, pr, pe, pres_d, ctx["active"], ctx["queries"],
                                 seed=s)
            p = eval_probe(pr, pres_d, ctx["active"], ctx["label"], s)

            MRR.append(r["micro_by_transform"]["raw"]["MRR"])
            MRRC.append(r["micro_by_transform"]["center"]["MRR"])
            PMRK.append(diag["patchmean_eff_rank"])
            ACC.append(p["acc"])
            PVAL.append(r["micro"]["p_value_vs_null"])
            secs.append(tm["train_sec"])
            QRK.append(float(np.nanmean([v["query_eff_rank"]
                                         for v in r["bank_stats"].values()])))
            SELF.append(float(np.nanmean([v["query_self_similarity"]
                                          for v in r["bank_stats"].values()])))
            MARG.append(float(np.nanmean([v.get("gold_margin_z", np.nan)
                                          for v in r["bank_stats"].values()])))

            print(f"  [neg {nm:<15s}] seed {s} | probe {p['acc']:.3f} "
                  f"| MRR raw {MRR[-1]:.5f} centred {MRRC[-1]:.5f} "
                  f"| pm_rk {diag['patchmean_eff_rank']:6.2f} "
                  f"| q_rk {QRK[-1]:5.2f} self-sim {SELF[-1]:+.3f} "
                  f"| margin_z {MARG[-1]:+.3f} | p={PVAL[-1]:.3f} "
                  f"| {tm['train_sec']:.0f}s", flush=True)

            if si == 0:
                cand_art = {"spec": spectrum_of(diag["pooled_flat_ref"])[0],
                            "pca": pca2_of(diag["pooled_flat_ref"]),
                            "ranks": r["ranks_by_transform"]["raw"][:RANK_KEEP],
                            "hist": np.asarray(hist, dtype=np.float32)}
            del model, pooler, pr, diag
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not MRR:
            continue

        row = {"neg_mode": nm,
               "MRR": _mm(MRR), "MRR_center": _mm(MRRC),
               "pm_rk": _mm(PMRK), "query_rk": _mm(QRK),
               "query_self_sim": _mm(SELF), "gold_margin_z": _mm(MARG),
               "probe_acc": _mm(ACC), "p_value_max": float(np.nanmax(PVAL)),
               "train_sec_mean": float(np.mean(secs)),
               "same_aspect_negatives": nm in ("aspect_matched", "aspect_hard",
                                               "fullpool"),
               "n_negatives": (args.neg_pool if nm in ("fullpool", "aspect_hard")
                               else (INFONCE_BATCH - 1 if nm != "cos" else 0))}
        rows.append(row)
        ART["scatter"].append({"name": f"neg:{nm}", "pm_rk": row["pm_rk"][0],
                               "mrr": row["MRR"][0], "probe": row["probe_acc"][0],
                               "kind": "negatives"})
        if best is None or row["MRR"][0] > best["MRR"][0]:
            best, best_art = row, cand_art

    if best_art:
        if best_art.get("spec") is not None:
            ART["spectra"]["pooled_neg_best"] = best_art["spec"]
        if best_art.get("pca") is not None:
            ART["pca"]["pooled_neg_best"] = best_art["pca"]
        ART["ranks"]["neg_best"] = best_art["ranks"]
        if best_art["hist"].size:
            ART["history"]["neg_best"] = best_art["hist"]

    print("\n  ---- NEGATIVE-SAMPLING SWEEP (Protocol R) ----")
    for r in rows:
        flag = "same-aspect" if r["same_aspect_negatives"] else "aspect-mixed"
        print(f"    {r['neg_mode']:<16s} {flag:<12s} negs={r['n_negatives']:<5d} "
              f"| MRR {r['MRR'][0]:.5f} ± {r['MRR'][1]:.5f} "
              f"| centred {r['MRR_center'][0]:.5f} "
              f"| q_rk {r['query_rk'][0]:6.2f} "
              f"| probe {r['probe_acc'][0]:.3f} | p={r['p_value_max']:.3f}")

    base = next((r for r in rows if r["neg_mode"] == "cos"), None)
    verdict_txt = None
    if best and base:
        gain = best["MRR"][0] / max(base["MRR"][0], 1e-12)
        print(f"    >>> best: {best['neg_mode']} ({gain:.1f}x over the cosine baseline)")
        # a real fix must ALSO clear the null, not merely beat a chance-level baseline
        cleared = best["p_value_max"] < 0.05
        if gain > 10 and cleared:
            verdict_txt = "negatives_are_the_fix"
            print("    >>> NEGATIVES ARE THE FIX: the objective, not the architecture, "
                  "was the binding constraint. Design rule: draw in-batch negatives "
                  "from WITHIN the dominant categorical partition.")
        elif cleared:
            verdict_txt = "partial"
            print("    >>> PARTIAL: above the null but far below the oracle. Report "
                  "bits carried, not the ratio.")
        else:
            verdict_txt = "null_survives"
            print("    >>> negatives do NOT repair it: every regime remains "
                  "indistinguishable from a random ranker (p >= 0.05). The null "
                  "survives the strongest available objective.")

        # the informative contrast for the paper: mixed vs matched negatives
        mixed = next((r for r in rows if r["neg_mode"] == "inbatch_random"), None)
        matched = next((r for r in rows if r["neg_mode"] == "aspect_matched"), None)
        if mixed and matched:
            print(f"    >>> shortcut test: aspect-mixed {mixed['MRR'][0]:.5f} "
                  f"(probe {mixed['probe_acc'][0]:.3f}) vs aspect-matched "
                  f"{matched['MRR'][0]:.5f} (probe {matched['probe_acc'][0]:.3f})")
            if matched["probe_acc"][0] < mixed["probe_acc"][0] - 0.05:
                print("        the probe DROPS under matched negatives -> the shortcut "
                      "was indeed what the probe was measuring.")

    return {"rows": rows, "best": best, "seeds": seeds, "verdict": verdict_txt,
            "neg_pool": args.neg_pool, "hard_k": NEG_HARD_K}


def stage_oracle(ctx, args):
    out = {}
    data, active, mem = ctx["data"], ctx["active"], ctx["membership"]
    raw_x = ctx.get("raw_x")
    paper_x = ctx.get("paper_x")
    if paper_x is None and "paper" in data.node_types:
        paper_x = data["paper"].x.detach().float().cpu()

    for feat_tag, feats in (("raw", None), ("whitened", ctx["whitened_x"])):
        asp = aspect_feature_matrix(data, active, mem, feats=feats, raw_x=raw_x)
        if feats is None:
            pv = paper_x
        elif "paper" in feats:
            pv = feats["paper"].detach().float()
        else:
            pv = None
            print(f"  [note] whitened features lack 'paper' -> with_summary == "
                  f"no_summary for feats={feat_tag}")
        ws = oracle_protocol_r(asp, ctx["pres"], active, ctx["queries"],
                               include_paper=pv, tag=f"oracle-{feat_tag}")
        ns = oracle_protocol_r(asp, ctx["pres"], active, ctx["queries"],
                               include_paper=None, tag=f"oracle-{feat_tag}-nosum")
        if feat_tag == "whitened":
            ART["ranks"]["oracle"] = ws["_ranks"][:RANK_KEEP]
            flat = asp.reshape(-1, asp.size(-1))
            sp, _ = spectrum_of(flat)
            if sp is not None:
                ART["spectra"]["oracle_features"] = sp
            ART["pca"]["oracle_features"] = pca2_of(flat)
            ART["scatter"].append({"name": "oracle",
                                   "pm_rk": float(eff_rank_only(_subsample(flat, SPEC_ROWS))),
                                   "mrr": ws["micro"]["MRR"], "probe": None,
                                   "kind": "oracle"})
        ws.pop("_ranks", None); ns.pop("_ranks", None)
        out[f"{feat_tag}_with_summary"], out[f"{feat_tag}_no_summary"] = ws, ns
        for a in active:
            w = ws.get(a, {})
            print(f"  [oracle:{feat_tag}] {a:<7s} ctx={_fmt(w.get('MRR'))} "
                  f"centred={_fmt(w.get('MRR_center'))} "
                  f"no-summary={_fmt(ns.get(a, {}).get('MRR'))} "
                  f"cheat={_fmt(w.get('cheat_MRR'))} "
                  f"dc={_fmt((w.get('dc') or {}).get('dc_ratio'), 2)} "
                  f"erank={_fmt(w.get('eff_rank'), 1)}")
        del asp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vals = [v["MRR"] for k in out if k.endswith("with_summary")
            for a, v in out[k].items() if a != "micro" and isinstance(v, dict)
            and not math.isnan(v["MRR"])]
    out["ctx_range"] = (min(vals), max(vals)) if vals else (None, None)

    out["bm25_micro"] = out["nooverlap_micro"] = None
    if args.text_controls:
        try:
            texts, files = load_texts(RAW_DIR, active)
            if len(files) < ctx["P"]:
                raise RuntimeError(f"texts({len(files)}) < papers({ctx['P']})")
            bm, nov = {}, {}
            pres_np = ctx["pres"].cpu().numpy()
            for i, a in enumerate(active):
                q_ids = ctx["queries"][a]
                if q_ids.numel() < 2:
                    continue
                if sum(1 for t in texts[a] if t) < 0.5 * len(files):
                    print(f"  [skip] aspect '{a}': text mostly empty"); continue
                cand_ids = np.nonzero(pres_np[:, i])[0]
                pos = {int(p): j for j, p in enumerate(cand_ids)}
                q = [int(p) for p in q_ids.tolist() if int(p) in pos]
                cand_txt = [texts[a][p] for p in cand_ids]
                others = [b for b in active if b != a]
                q_txt = [" ".join([texts[b][p] for b in others] + [texts["paper"][p]])
                         for p in q]
                gold = np.array([pos[p] for p in q])
                bm[a] = _metrics_from_ranks(
                    torch.from_numpy(bm25_retrieval(q_txt, cand_txt, gold)),
                    len(cand_ids), f"bm25/{a}")
                print(f"  [bm25] {a:<7s} MRR={bm[a]['MRR']:.4f}")
                scrub = [strip_shared_ngrams(t, texts[a][p], NGRAM_N)
                         for t, p in zip(q_txt, q)]
                qe = sbert_embed(scrub).to(DEVICE)
                ce = sbert_embed(cand_txt).to(DEVICE)
                rr = rank_against_full_pool(qe, ce, torch.from_numpy(gold).long())
                nov[a] = _metrics_from_ranks(rr, len(cand_ids), f"nooverlap/{a}")
                print(f"  [no-overlap] {a:<7s} MRR={nov[a]['MRR']:.4f}")
                del qe, ce
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if bm:
                out["bm25"], out["no_overlap"] = bm, nov
                out["bm25_micro"] = _pool_micro(bm)["MRR"]
                out["nooverlap_micro"] = _pool_micro(nov)["MRR"]
        except Exception as e:                              # noqa: BLE001
            print(f"  [warn] text controls skipped: {e}")
    return out


def stage_fix(ctx, args):
    grid = [("mean", "cos", False), ("mean", "infonce", False), ("sum", "cos", False),
            ("deepsets", "cos", False), ("deepsets", "infonce", False),
            ("attn", "infonce", False), ("mean", "cos", True)]
    rows, best, best_art = [], None, None
    for pool_mode, loss_mode, tv in grid:
        MRR, MRRC, PMRK, ACC, H10, secs = [], [], [], [], [], []
        cand_art = None
        for si, s in enumerate(args.seeds):
            try:
                model, pooler, pr, pe, pres_d, mask_idx, diag, tm, hist = train_variant(
                    s, ctx["data"], ctx["active"], ctx["membership"], ctx["pres"],
                    ctx["rwse"], ctx["maskable"], ctx["pe_dim"], pool_mode=pool_mode,
                    loss_mode=loss_mode,
                    whitened_x=ctx["whitened_x"] if args.fix_whiten else None,
                    epochs=args.epochs, tgt_vic=tv, track=(si == 0))
            except Exception as e:                          # noqa: BLE001
                print(f"  [ERROR] {pool_mode}/{loss_mode}/vic={tv} seed {s}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            r = protocol_r_model(model, pr, pe, pres_d, ctx["active"], ctx["queries"],
                                 seed=s)
            p = eval_probe(pr, pres_d, ctx["active"], ctx["label"], s)
            MRR.append(r["micro_by_transform"]["raw"]["MRR"])
            MRRC.append(r["micro_by_transform"]["center"]["MRR"])
            H10.append(r["micro"]["Hits@10"])
            PMRK.append(diag["patchmean_eff_rank"]); ACC.append(p["acc"])
            secs.append(tm["train_sec"])
            print(f"  [fix {pool_mode}/{loss_mode}{'/tgtvic' if tv else ''}] seed {s} "
                  f"| probe {p['acc']:.3f} | MRR raw {MRR[-1]:.5f} centred {MRRC[-1]:.5f} "
                  f"| pm_rk {diag['patchmean_eff_rank']:.2f} | {tm['train_sec']:.0f}s",
                  flush=True)
            if si == 0:
                cand_art = {"hist": np.asarray(hist, dtype=np.float32),
                            "spec": spectrum_of(diag["pooled_flat_ref"])[0],
                            "pca": pca2_of(diag["pooled_flat_ref"]),
                            "ranks": r["ranks_by_transform"]["raw"][:RANK_KEEP]}
            del model, pooler, pr, diag
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not MRR:
            continue
        row = {"pooling": pool_mode, "loss": loss_mode, "target_vic": tv,
               "MRR": _mm(MRR), "MRR_center": _mm(MRRC), "Hits@10": _mm(H10),
               "pm_rk": _mm(PMRK), "probe_acc": _mm(ACC),
               "train_sec_mean": float(np.mean(secs))}
        rows.append(row)
        ART["scatter"].append({"name": f"{pool_mode}/{loss_mode}" + ("+vic" if tv else ""),
                               "pm_rk": row["pm_rk"][0], "mrr": row["MRR"][0],
                               "probe": row["probe_acc"][0], "kind": "fix"})
        if best is None or row["MRR"][0] > best["MRR"][0]:
            best, best_art = row, cand_art
    if best_art:
        ART["history"]["best_fix"] = best_art["hist"]
        if best_art["spec"] is not None:
            ART["spectra"]["pooled_fix"] = best_art["spec"]
        ART["pca"]["pooled_fix"] = best_art["pca"]
        ART["ranks"]["fix"] = best_art["ranks"]
    print("\n  ---- FIX SUMMARY (Protocol R) ----")
    for r in rows:
        print(f"    {r['pooling']:<9s} {r['loss']:<8s}"
              f"{'+vic' if r['target_vic'] else '    '} "
              f"| pm_rk {r['pm_rk'][0]:6.2f} | MRR {r['MRR'][0]:.5f} ± {r['MRR'][1]:.5f} "
              f"| centred {r['MRR_center'][0]:.5f}")
    return {"rows": rows, "best": best, "best_mrr": (best["MRR"][0] if best else None)}


def stage_poolsize(ctx, args):
    """
    Free experiment: exact MRR vs candidate-pool size for every stored rank
    vector, by hypergeometric thinning. Converts "at chance on 58k" into a
    capacity statement in bits.
    """
    n_full = int(ctx["P"])
    rng = np.random.default_rng(0)
    curves, bits = {}, {}
    for k, v in ART["ranks"].items():
        v = np.asarray(v)
        if v.size < 10:
            continue
        rows = []
        for n in POOLSIZE_GRID + [n_full]:
            if n > n_full:
                continue
            m = mrr_at_pool_size(v, n_full, n, rng)
            rows.append({"N": int(n), "MRR": m, "chance": chance_mrr(n)})
        curves[k] = rows
        bits[k] = bits_carried(rows[-1]["MRR"], n_full)
    need = math.log2(n_full)
    print("  ---- CAPACITY (exact pool-size thinning) ----")
    print(f"    identifying one of {n_full} papers requires {need:.1f} bits")
    for k in curves:
        last = curves[k][-1]
        print(f"    {k:<18s} MRR@{n_full}={last['MRR']:.5f} "
              f"| carries {bits[k]:5.2f} of {need:.1f} bits "
              f"({100*bits[k]/need:5.1f}%)")
    return {"curves": curves, "bits_carried": bits, "bits_required": need,
            "n_full": n_full}


def stage_rho(ctx, args):
    """
    THE SEVERITY LAW (redesigned in v3.1).

    v3's sweep queried with the TRUE target vector, so it measured candidate-bank
    discriminability and returned MRR=1.0 for every rho. It could not, even in
    principle, exhibit predictor collapse. This version runs two sub-experiments:

      (1) severity law     ridge ceiling (no budget) vs a budgeted MLP trained
                           with the cosine JEPA loss, on identical data. The gap
                           IS the mechanism.
      (2) negative control the old sweep, correctly relabelled. Establishes that
                           high rho does not by itself impair separability.
    """
    print("  ---- (1) SEVERITY LAW: ridge ceiling vs budgeted predictor ----")
    rows_tr, thr = rho_sweep_trained(seed=0)

    print("\n  ---- (2) NEGATIVE CONTROL: true-context query, no learning ----")
    rows_geo = rho_sweep_geometry(seed=0)

    meas = measured_redundancy(ctx, args.res_ref)
    legacy = float("nan")
    try:
        legacy = estimate_rho_real(ctx["data"], ctx["active"], ctx["membership"],
                                   raw_x=ctx.get("raw_x"))
    except Exception as e:                                  # noqa: BLE001
        print(f"  [warn] legacy rho_hat failed: {e}")

    pv = ((args.res_ref.get("protocolr") or {})
          .get("euclidean_mean_raw") or {}).get("pooled_variance")
    corpus_rho = pv["between_frac"] if pv else None
    verdict_txt = None
    if corpus_rho:
        near = min(rows_tr, key=lambda r: abs(r["rho"] - corpus_rho))
        print(f"\n  [rho] corpus operating point : rho={corpus_rho:.4f} "
              f"(between/within = {corpus_rho/max(1-corpus_rho,1e-9):.1f})")
        print(f"  [rho] nearest synthetic      : m={near['m']} rho={near['rho']:.4f} "
              f"| ridge={near['MRR_ridge']:.4f} trained={near['MRR_trained']:.4f} "
              f"({near['gap']:.0f}x gap)")
        if near["gap"] > 10:
            verdict_txt = "confirmed"
            print("    >>> SEVERITY LAW CONFIRMED: at the corpus rho, instance "
                  "identity is linearly recoverable (ridge) but a budgeted "
                  "optimiser does not recover it. This reproduces oracle >> "
                  "trained in silico, on data with no graph, no pooling and no "
                  "encoder.")
        else:
            verdict_txt = "not_reproduced"
            print("    >>> LAW NOT REPRODUCED at this budget. Do NOT present rho as "
                  "a result -- state it as a hypothesis in the discussion and say "
                  "so explicitly in Limitations.")
    print(f"  [rho] collapse frontier (trained < 50% of ridge): {thr}")

    # sanity: the generator should reproduce the requested rho
    bad = [r for r in rows_tr if abs(r["between_frac_measured"] - r["rho"]) > 0.05]
    if bad:
        print(f"  [warn] {len(bad)}/{len(rows_tr)} synthetic rows deviate >0.05 from "
              f"the requested rho -- check _rho_make_task normalisation")

    return {"rows_trained": rows_tr, "rows_geometry": rows_geo,
            "collapse_frontier": thr,          # STRING keys ("m=1", "m=4")
            "measured": meas, "corpus_rho": corpus_rho, "verdict": verdict_txt,
            "legacy_intra_patch_cosine": legacy,
            "config": {"steps": RHO_STEPS, "batch": RHO_BATCH, "lr": RHO_LR,
                       "n": RHO_N, "dim": RHO_DIM, "id_dim": RHO_ID_DIM,
                       "snr": RHO_SNR, "aspects": RHO_ASPECTS,
                       "fit_frac": RHO_FIT_FRAC},
            "semantics": "rho = between-aspect variance fraction of the target"}


# ============================================================================
# 13. COST
# ============================================================================
def build_cost(phases, wall_sec):
    wh = wall_sec / 3600.0
    gh = wh * NUM_GPUS_BILLED
    return {"hardware": {"device": str(DEVICE),
                         "gpu_name": torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else "cpu",
                         "num_gpus_billed": NUM_GPUS_BILLED,
                         "platform": platform.platform()},
            "rate": {"gpu_hourly_rate": GPU_HOURLY_RATE_USD,
                     "currency": COST_CURRENCY},
            "measured_seconds": {"wall_total": wall_sec,
                                 "phase_breakdown": phases},
            "gpu_hours": {"wall_hours": wh, "gpu_hours_billed": gh},
            "peak_gpu_gb": max([p["peak_gpu_gb"] for p in phases] + [0.0]),
            "estimated_cost": {"amount": round(gh * GPU_HOURLY_RATE_USD, 4),
                               "currency": COST_CURRENCY,
                               "formula": "wall_hours * num_gpus_billed * rate"}}


def print_cost(cs):
    print("\n" + "=" * 90 + "\n  COST / GPU-HOUR SUMMARY (measured this run)\n" + "=" * 90)
    print(f"  GPU              : {cs['hardware']['gpu_name']} "
          f"x{cs['hardware']['num_gpus_billed']}")
    print(f"  Peak GPU memory  : {cs['peak_gpu_gb']:.2f} GB")
    for p in cs["measured_seconds"]["phase_breakdown"]:
        print(f"    {p['phase']:<44s} {_fmt_hms(p['seconds'])}")
    print(f"  WALL TOTAL       : {_fmt_hms(cs['measured_seconds']['wall_total'])} "
          f"({cs['gpu_hours']['wall_hours']:.2f} h)")
    print(f"  GPU-HOURS        : {cs['gpu_hours']['gpu_hours_billed']:.2f}")
    print(f"  EST. COST        : {cs['estimated_cost']['amount']:.2f} "
          f"{cs['estimated_cost']['currency']}")
    print("=" * 90)


# ============================================================================
# 14. HEADLINE SUMMARY
# ============================================================================
def print_headline(res, P):
    print("\n" + "=" * 90 + "\n  HEADLINE SUMMARY\n" + "=" * 90)
    ch = chance_mrr(P)

    base = ((res.get("protocolr") or {}).get("euclidean_mean_raw") or {}).get("micro", {})
    if base:
        print(f"  trained baseline MRR  : {base.get('MRR', float('nan')):.5f} "
              f"(chance {ch:.5f}, p={base.get('p_value_vs_null', float('nan')):.3f})")
        if base.get("p_value_vs_null", 0) > 0.05:
            print("    -> NOT distinguishable from a random ranker. Report the null "
                  "test, not a ratio against chance.")

    pv = ((res.get("protocolr") or {}).get("euclidean_mean_raw") or {}).get("pooled_variance")
    if pv:
        print(f"  variance split        : {100*pv['between_frac']:.2f}% between ASPECT, "
              f"{100*pv['within_frac']:.2f}% between PAPER")
        print("    -> the objective is satisfied by the aspect component alone.")

    orc = ((res.get("oracle") or {}).get("ctx_range") or (None, None))
    if orc[0]:
        print(f"  training-free oracle  : {orc[0]:.3f} - {orc[1]:.3f}")
        if base and base.get("MRR") and orc[0] > base["MRR"]:
            print(f"    -> training is {orc[0]/max(base['MRR'],1e-12):.0f}x WORSE than "
                  f"not training, on identical features and code.")

    ce = (res.get("centering") or {}).get("by_transform")
    if ce:
        b = max(ce, key=lambda t: ce[t]["MRR"][0])
        gain = ce[b]["MRR"][0] / max(ce["raw"]["MRR"][0], 1e-12)
        print(f"  best retrieval frame  : {b} -> {ce[b]['MRR'][0]:.5f} ({gain:.1f}x raw)")
        print("    -> " + ("FRAME MATTERS: the signal was present but DC-dominated."
                           if gain > 5 else
                           "frame does not help: the query itself is degenerate."))

    pa = res.get("peraspect") or {}
    single = [a for a, v in pa.items()
              if isinstance(v, dict) and v.get("members_per_patch", 9) <= 1.01]
    for a in single:
        v = pa[a]
        print(f"  pooling control ({a:<6s}): m=1.00 -> pooling IS the identity map; "
              f"pool_rk {v['pool_rk'][0]:.1f} vs query_rk {v['query_rk'][0]:.2f}, "
              f"margin_z {v['margin_z'][0]:+.3f}")

    lat = (res.get("latent") or {}).get("taps")
    if lat:
        pr_ = lat.get("pred", {}).get("micro", {}).get("MRR", float("nan"))
        pe_ = lat.get("pe_only", {}).get("micro", {}).get("MRR", float("nan"))
        bp = (res.get("latent") or {}).get("best_probe_tap")
        bpv = (lat.get(bp, {}).get("probe_micro", {}).get("MRR", float("nan"))
               if bp else float("nan"))
        print(f"  information ladder    : pred {pr_:.5f} vs pe_only {pe_:.5f}")
        if np.isfinite(pr_) and np.isfinite(pe_) and abs(pr_ - pe_) < 0.25 * max(pr_, 1e-12):
            print("    -> QUERY-SIDE COLLAPSE CONFIRMED: the query is a function of the "
                  "aspect code alone.")
        print(f"  best linear read-out  : {bp} -> {bpv:.4f} (ridge onto oracle space)")
        if np.isfinite(bpv) and orc[0] and bpv > 0.25 * orc[0]:
            print("    -> identity IS still present upstream: the loss discards it, the "
                  "encoder does not destroy it.")
        elif np.isfinite(bpv):
            print("    -> identity is absent at every tap: the encoder/target branch "
                  "destroys it before the loss can use it.")

    ng = (res.get("negatives") or {})
    if ng.get("best") and ng.get("rows"):
        b0 = next((r for r in ng["rows"] if r["neg_mode"] == "cos"), None)
        if b0:
            g = ng["best"]["MRR"][0] / max(b0["MRR"][0], 1e-12)
            print(f"  best negative regime  : {ng['best']['neg_mode']} "
                  f"-> {ng['best']['MRR'][0]:.5f} ({g:.1f}x cosine, "
                  f"p={ng['best']['p_value_max']:.3f})")
            v = ng.get("verdict")
            print("    -> " + {"negatives_are_the_fix":
                               "NEGATIVES ARE THE FIX -> the paper ships a design rule.",
                               "partial":
                               "partial recovery; report bits carried.",
                               "null_survives":
                               "negatives do not repair it; the null is airtight."}
                  .get(v, "inconclusive."))

    en = (res.get("encoder") or {}).get("best")
    if en:
        rows = (res["encoder"] or {}).get("rows", [])
        d0 = [r for r in rows if r["depth"] == 0]
        d3 = [r for r in rows if r["depth"] == 3]
        print(f"  best encoder variant  : {en['tag']} MRR {en['MRR'][0]:.5f} "
              f"node_rk {en['node_rk'][0]:.1f}")
        if d0 and d3:
            print(f"    depth 0 {d0[0]['MRR'][0]:.5f} vs depth 3 {d3[0]['MRR'][0]:.5f} -> " +
                  ("ENCODER IS THE BOTTLENECK (over-smoothing)."
                   if d0[0]["MRR"][0] > 5 * max(d3[0]["MRR"][0], 1e-12)
                   else "depth is not the binding factor."))

    rh = res.get("rho") or {}
    if rh.get("rows_trained"):
        cr = rh.get("corpus_rho")
        if cr:
            near = min(rh["rows_trained"], key=lambda r: abs(r["rho"] - cr))
            print(f"  severity law (synth)  : at rho={near['rho']:.4f}  "
                  f"ridge {near['MRR_ridge']:.4f} vs trained {near['MRR_trained']:.4f} "
                  f"({near['gap']:.0f}x)")
            print("    -> " + {"confirmed":
                               "CONFIRMED in silico with no graph, pooling or encoder.",
                               "not_reproduced":
                               "not reproduced at this budget -> demote to hypothesis."}
                  .get(rh.get("verdict"), "inconclusive."))
        print(f"  collapse frontier     : {rh.get('collapse_frontier')}")

    ps = res.get("poolsize") or {}
    if ps.get("bits_carried"):
        need = ps["bits_required"]
        bl = ps["bits_carried"].get("baseline")
        orb = ps["bits_carried"].get("oracle")
        if bl is not None:
            print(f"  capacity              : baseline carries {bl:.2f} of "
                  f"{need:.1f} bits" +
                  (f"; oracle carries {orb:.2f}" if orb is not None else ""))
    print("=" * 90)


# ============================================================================
# 15. MAIN
# ============================================================================
STAGE_ORDER = ["rankinit", "protocolr", "peraspect", "centering", "encoder",
               "latent", "negatives", "oracle", "fix", "poolsize", "rho"]

# Stages that only READ artefacts/results produced by earlier stages.
DEPENDENT_STAGES = {
    "poolsize": "rank vectors in ART (protocolr / oracle / negatives)",
    "rho":      "protocolr or centering DC / variance numbers",
}


def parse_args():
    ap = argparse.ArgumentParser("TODO experiments v3.1 + figures")
    ap.add_argument("--stages", default="all",
                    help=f"comma list of {STAGE_ORDER} or 'all'")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEF)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEF))
    ap.add_argument("--n-queries", type=int, default=N_QUERIES)
    ap.add_argument("--text-controls", action="store_true")
    ap.add_argument("--fix-whiten", action="store_true")
    ap.add_argument("--enc-whiten", action="store_true")
    ap.add_argument("--enc-loss", default="cos", choices=["cos", "infonce"])
    ap.add_argument("--enc-max-seeds", type=int, default=ENC_MAX_SEEDS)
    ap.add_argument("--centering-max-seeds", type=int, default=3)
    ap.add_argument("--latent-max-seeds", type=int, default=2)
    ap.add_argument("--neg-max-seeds", type=int, default=3)
    ap.add_argument("--neg-pool", type=int, default=NEG_POOL)
    ap.add_argument("--rho-steps", type=int, default=RHO_STEPS,
                    help="optimisation budget for the synthetic severity law")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="merge into an existing results.json / artifacts.npz "
                         "instead of starting from scratch (lets you add one "
                         "stage without re-running the sweep)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    want = STAGE_ORDER if a.stages.strip() == "all" else \
        [s.strip() for s in a.stages.split(",") if s.strip()]
    a.stages = [s for s in STAGE_ORDER if s in want] + \
               [s for s in want if s not in STAGE_ORDER]
    a.seeds = [int(s) for s in str(a.seeds).split(",") if str(s).strip()]
    if not a.seeds:
        a.seeds = [0]
    return a


def main():
    args = parse_args()
    global OUT_DIR, FIG_DIR, N_QUERIES, RHO_STEPS
    OUT_DIR = args.out_dir
    FIG_DIR = os.path.join(OUT_DIR, "figures")
    N_QUERIES = args.n_queries
    RHO_STEPS = args.rho_steps
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    res_path = os.path.join(OUT_DIR, "todo_experiments_results.json")
    art_path = os.path.join(OUT_DIR, "artifacts.npz")

    # ---------------- figures-only fast path ----------------
    if args.figures_only:
        if not os.path.exists(res_path):
            raise FileNotFoundError(res_path)
        res = json.load(open(res_path))
        load_artifacts(art_path)
        make_all_figures(res)
        print(f"\n[done] figures re-rendered into {FIG_DIR}")
        return

    wall0, phases = time.time(), []
    print("=" * 100)
    print("  TODO EXPERIMENTS v3.1  |  stages =", args.stages)
    print(f"  epochs={args.epochs} seeds={args.seeds} n_queries={N_QUERIES} "
          f"text_controls={args.text_controls} enc_loss={args.enc_loss} "
          f"neg_pool={args.neg_pool} rho_steps={RHO_STEPS} resume={args.resume}")
    print(f"  device={DEVICE} gpu="
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print("=" * 100)

    # ---------------- inputs ----------------
    if not os.path.exists(RWSE_CACHE):
        raise FileNotFoundError(f"RWSE cache missing: {RWSE_CACHE} (run train.build_rwse)")
    blob = torch.load(RWSE_CACHE, weights_only=False)
    rwse, pe_dim, active = blob["patch_rwse"], blob["rwse_steps"], blob["active"]

    with Timer("build_graph", phases):
        data, _ = build_hetero_graph(RAW_DIR, CACHE_PATH, rebuild=False)
        data = T.ToUndirected()(data)
    patch_idx = build_patch_index(data, active)
    pres = aspect_presence(data, patch_idx, active)
    label = build_reasoning_label(data)
    maskable = torch.nonzero(pres.sum(1) >= 2, as_tuple=False).squeeze(1)
    P = data["paper"].num_nodes
    print(f"  active={active} | P={P} | maskable={maskable.numel()}")

    with Timer("membership+whiten", phases):
        membership = build_membership(data, active)
        counts = membership_counts(membership, active, P)
        whitened_x = whiten_features(data, active)
        snap_types = list(active) + (["paper"] if "paper" not in active else [])
        raw_x = {nt: data[nt].x.detach().float().cpu().clone()
                 for nt in snap_types
                 if nt in data.node_types and getattr(data[nt], "x", None) is not None}
        paper_x_cpu = raw_x.get("paper")
        print(f"  [snapshot] raw features cached on CPU for {list(raw_x)} "
              f"({sum(v.numel() for v in raw_x.values()) * 4 / 1e9:.2f} GB)")

    for a in active:
        print(f"  [cardinality] {a:<8s} members/paper mean={counts[a].float().mean():.2f} "
              f"max={int(counts[a].max())}")

    queries = build_query_set(pres, maskable, active, n_queries=N_QUERIES, seed=QUERY_SEED)
    print(f"  [protocolR] chance MRR at full pool (P={P}) = {chance_mrr(P):.3e}")
    print(f"  [capacity]  identifying 1 of {P} requires {math.log2(P):.1f} bits")

    ctx = dict(data=data, active=active, membership=membership, counts=counts, pres=pres,
               label=label, rwse=rwse, maskable=maskable, pe_dim=pe_dim,
               whitened_x=whitened_x, queries=queries, P=P,
               raw_x=raw_x, paper_x=paper_x_cpu)

    # ---------------- results skeleton (optionally resumed) ----------------
    res = {}
    if args.resume and os.path.exists(res_path):
        try:
            res = json.load(open(res_path))
            load_artifacts(art_path)
            print(f"  [resume] merged previous results from {res_path} "
                  f"(stages present: {[k for k in res if k not in ('config', 'cost')]})")
        except Exception as e:                              # noqa: BLE001
            print(f"  [warn] resume failed ({e}); starting fresh")
            res = {}

    res["config"] = {"stages": args.stages, "epochs": args.epochs, "seeds": args.seeds,
                     "n_queries": N_QUERIES, "protocol": "R (full aspect-matched pool)",
                     "transforms": list(CENTERING_TRANSFORMS),
                     "taps": list(LATENT_TAPS), "neg_modes": list(NEG_MODES),
                     "chance_MRR": chance_mrr(P), "bits_required": math.log2(P),
                     "active": active, "enc_hidden": ENC_HIDDEN,
                     "enc_loss": args.enc_loss, "neg_pool": args.neg_pool,
                     "rho_steps": RHO_STEPS, "rho_grid": list(RHO_GRID),
                     "latent_taps_ridge_lambda": RIDGE_LAMBDA,
                     "bootstrap_resamples": BOOT_N, "permutations": N_PERM,
                     "maskable": int(maskable.numel()), "P": int(P),
                     "version": "v3.1"}
    args.res_ref = res      # later stages read earlier stages' numbers

    # ---------------- stage dispatch ----------------
    fns = {"rankinit":  stage_rankinit,
           "protocolr": stage_protocolr,
           "peraspect": stage_peraspect,
           "centering": stage_centering,
           "encoder":   stage_encoder,
           "latent":    stage_latent,
           "negatives": stage_negatives,
           "oracle":    stage_oracle,
           "fix":       stage_fix,
           "poolsize":  stage_poolsize,
           "rho":       stage_rho}

    failed = []
    for st in args.stages:
        if st not in fns:
            print(f"  [warn] unknown stage '{st}' -- skipped"); continue
        # dependency hints (never fatal: partial runs must still produce output)
        if st == "poolsize" and not ART["ranks"]:
            print(f"  [warn] stage 'poolsize' needs {DEPENDENT_STAGES['poolsize']}; "
                  "none cached. It will produce an empty curve set. Use --resume.")
        if st == "rho" and "protocolr" not in res and "centering" not in res:
            print(f"  [warn] stage 'rho' needs {DEPENDENT_STAGES['rho']}; none found. "
                  "The corpus operating point will not be marked. Use --resume.")

        print("\n" + "#" * 100 + f"\n#  STAGE: {st.upper()}\n" + "#" * 100)
        try:
            with Timer(f"stage_{st}", phases):
                res[st] = fns[st](ctx, args)
        except KeyboardInterrupt:
            print("\n  [abort] interrupted by user -- saving partial results")
            break
        except Exception:                                   # noqa: BLE001
            import traceback; traceback.print_exc()
            res[st] = {"error": "see traceback in the log"}
            failed.append(st)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------- headline + cost ----------------
    try:
        print_headline(res, P)
    except Exception as e:                                  # noqa: BLE001
        print(f"  [warn] headline summary failed: {e}")

    res["cost"] = build_cost(phases, time.time() - wall0)
    print_cost(res["cost"])

    if failed:
        print(f"\n  [warn] stages that errored: {failed} "
              f"(their entries in the JSON carry an 'error' key)")

    # ---------------- persist ----------------
    def _jsonable(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if torch.is_tensor(o):
            return o.detach().cpu().tolist()
        return float(o)

    # FIX(v3.1): keys are not necessarily strings (stage_rho used int keys in v3),
    # and json.dump cannot serialise numpy scalar keys.
    def _strip_arrays(node):
        if isinstance(node, dict):
            return {(k.item() if isinstance(k, (np.integer, np.floating)) else k):
                        _strip_arrays(v)
                    for k, v in node.items()
                    if not (isinstance(v, np.ndarray) and v.size > 64)
                    and not (isinstance(k, str) and k.startswith("_ranks"))}
        if isinstance(node, (list, tuple)):
            return [_strip_arrays(v) for v in node]
        return node

    res_clean = _strip_arrays(res)
    json.dump(res_clean, open(res_path, "w"), indent=2, default=_jsonable)
    print(f"\n[save] {res_path}")
    save_artifacts(art_path)

    # ---------------- figures ----------------
    if not args.no_figures:
        with Timer("figures", phases):
            make_all_figures(res_clean)

    print("\nRe-render figures later without re-training:")
    print(f"  python -m train.paper_reason_todo_experiments --figures-only "
          f"--out-dir {OUT_DIR}")
    print("Add a single stage to this result set without redoing the sweep:")
    print(f"  python -m train.paper_reason_todo_experiments --resume "
          f"--stages latent,negatives --out-dir {OUT_DIR}")


if __name__ == "__main__":
    main()
