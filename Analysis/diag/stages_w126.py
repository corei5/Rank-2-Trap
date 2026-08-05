"""Direct tests of W1 (causal mechanism), W2 (dissociation), W6 (capability).

Why this file exists
--------------------
The main suite could not adjudicate W1/W2/W6 because two substitutions happened
silently and simultaneously: features fell back from hetero node features to
MPNet text embeddings, and the trainer fell back from the paper's model to
diag.refmodel. Those are claims ABOUT THE MODEL, so a null result in a surrogate
configuration is uninformative in either direction.

This module therefore:

  1. Runs a 2x2 CELL GRID  {fallback, real} trainer x {text, hetero} features,
     so exactly one factor changes between comparable cells.
  2. VERIFIES provenance at runtime. A cell that asked for hetero features and
     got text, or asked for the real trainer and got the fallback, is marked
     status='provenance_mismatch' and emits NO verdict. Silence is the correct
     output when the premise is unmet.
  3. PRE-REGISTERS every threshold (see THRESHOLDS) and writes them to
     w126_prereg.json before computing anything.
  4. Tests W1 CAUSALLY BY INTERVENTION on the real features: the aspect-mean
     magnitude is scaled to sweep the between-aspect variance share, the model
     is retrained at each dose, and bits recovered are measured. This is a
     dose-response curve in the actual feature space, not a simulation.
  5. Tests W2 with a PAIRED bootstrap on per-query reciprocal-rank differences
     (the measurements share queries, so marginal CIs are the wrong test).
  6. Guards W6 against the degeneracy that made it vacuous: the winning control
     must differ from the model under test both in CONFIG and in VALUE.

Usage
-----
    python -m diag.stages_w126 --corpus papers --out-dir <dir> \
        --cells fallback+text,fallback+hetero,real+text,real+hetero
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from diag import adapters
from diag.common import (bits_recovered, device, mrr, ranks_from_banks,
                         recall_at, set_seed, spectral_decay_kappa,
                         variance_split)

# ============================================================== PRE-REGISTERED
# Declared before any result is seen. Do not edit after looking at the output.
THRESHOLDS: Dict[str, float] = {
    # "collapse" == recovers less than this many bits of paper identity
    "collapse_bits": 1.0,
    # context is judged to linearly encode identity above this many bits
    "probe_encodes_bits": 4.0,
    # a control demonstrates capability above this many bits
    "control_capable_bits": 1.0,
    # paired difference is called real only if the CI excludes 0 at this level
    "paired_alpha": 0.05,
    # two configs are "numerically identical" below this bits difference,
    # which is the degeneracy that made W6 vacuous
    "degenerate_bits_eps": 1e-3,
    # simulation is said to reproduce an observed collapse within this many bits
    "sim_match_bits": 1.0,
    # kappa is called load-bearing only if removing it moves bits by more
    "kappa_material_bits": 0.5,
}

# Control objectives for W6. If your adapters.train_jepa takes different config
# keys, this is the ONLY dict you need to change.
MODEL_UNDER_TEST = {"name": "l2_ema_paper", "loss": "l2", "ema": True}
CONTROL_SPECS = [
    {"name": "infonce_same_arch", "loss": "infonce", "ema": True},
    {"name": "cosine_same_arch",  "loss": "cosine",  "ema": True},
    {"name": "l2_no_ema",         "loss": "l2",      "ema": False},
]

ASPECT_ALIASES = {
    "claim":  ["claim", "claims", "assertion", "assertions", "hypothesis"],
    "method": ["method", "methods", "methodology", "methodological_details",
               "procedure", "approach"],
    "result": ["result", "results", "key_results", "finding", "findings",
               "outcome", "outcomes"],
}
PAPER_NODE_ALIASES = ["paper", "papers", "doc", "document", "article", "root"]


# ================================================================== statistics
def paired_bootstrap(a: np.ndarray, b: np.ndarray, B: int = 5000,
                     alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Bootstrap the PAIRED difference a-b over shared queries.

    a and b must be aligned per-query statistics (here: reciprocal ranks).
    Returns the observed mean difference, a percentile CI, and a two-sided
    bootstrap p-value for H0: mean difference = 0.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must align: {a.shape} vs {b.shape}")
    d = a - b
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"mean_diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "p": float(min(p, 1.0)), "n_pairs": int(n),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def _geometry(X: torch.Tensor) -> Dict[str, float]:
    """rho (both groupings), kappa, effective rank -- the parameters W1 needs."""
    n, A = X.shape[0], X.shape[1]
    Z = F.normalize(X.float(), dim=-1)
    flat = Z.reshape(n * A, -1)
    rho_aspect = variance_split(flat, torch.arange(A).repeat(n))["rho"]
    rho_paper = variance_split(flat, torch.arange(n).repeat_interleave(A))["rho"]
    sub = Z[:, 0, :]
    if sub.shape[0] > 8000:
        g = torch.Generator(device="cpu").manual_seed(0)
        sub = sub[torch.randperm(sub.shape[0], generator=g)[:8000]]
    kappa = spectral_decay_kappa(sub)
    s = torch.linalg.svdvals(sub - sub.mean(0, keepdim=True))
    eff_rank = float((s.sum() ** 2 / (s ** 2).sum()).item())
    return {"rho_between_aspect": float(rho_aspect),
            "rho_between_paper": float(rho_paper),
            "kappa": float(kappa), "effective_rank": eff_rank}


# ==================================================================== features
def _node_store(graph) -> Dict[str, Any]:
    """Normalise PyG HeteroData / dict-of-dicts / {'x_dict': ...} to a dict."""
    if hasattr(graph, "node_types"):                       # HeteroData
        return {nt: graph[nt] for nt in graph.node_types}
    if isinstance(graph, dict):
        if "x_dict" in graph and isinstance(graph["x_dict"], dict):
            return {k: {"x": v} for k, v in graph["x_dict"].items()}
        return {k: v for k, v in graph.items() if not isinstance(k, tuple)}
    raise TypeError(f"unrecognised graph container: {type(graph)}")


def _get_x(store_entry) -> Optional[torch.Tensor]:
    for key in ("x", "feat", "features", "h", "emb", "embedding"):
        if hasattr(store_entry, key):
            v = getattr(store_entry, key)
            if isinstance(v, torch.Tensor):
                return v
        if isinstance(store_entry, dict) and key in store_entry:
            v = store_entry[key]
            if isinstance(v, torch.Tensor):
                return v
    return None


def resolve_node_map(graph, aspects: Sequence[str]) -> Dict[str, str]:
    """Map each logical aspect onto an actual node type present in the graph.

    Override entirely with e.g.
        GJEPA_NODE_MAP="claim=claims,method=methods,result=key_results"
    This is the KeyError('claim') from the main suite, made explicit.
    """
    env = os.environ.get("GJEPA_NODE_MAP", "").strip()
    store = _node_store(graph)
    types = list(store.keys())
    if env:
        out = {}
        for part in env.split(","):
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
        missing = [v for v in out.values() if v not in types]
        if missing:
            raise KeyError(f"GJEPA_NODE_MAP points at absent node types "
                           f"{missing}; graph has {types}")
        return out

    norm = {t: str(t).strip().lower().replace("-", "_") for t in types}
    mapping: Dict[str, str] = {}
    for asp in aspects:
        cands = ASPECT_ALIASES.get(asp, [asp])
        hit = None
        for t, tl in norm.items():                          # exact alias first
            if tl in cands:
                hit = t; break
        if hit is None:                                     # then substring
            for t, tl in norm.items():
                if any(c in tl or tl in c for c in cands):
                    hit = t; break
        if hit is None:
            raise KeyError(
                f"cannot map aspect '{asp}' onto any node type. Graph has "
                f"{types}. Set GJEPA_NODE_MAP='{aspects[0]}=<type>,...'")
        mapping[asp] = hit
    return mapping


def _align_to_papers(x: torch.Tensor, ntype: str, graph, store,
                     n_papers: int) -> torch.Tensor:
    """One feature row per paper for this node type.

    Strategy, in order:
      1. node count already equals n_papers -> assume corpus order
      2. an explicit paper-index attribute -> scatter-mean by paper
      3. an edge type (paper -> ntype) -> mean-pool neighbours
    Anything else raises with a description of what was found.
    """
    if x.shape[0] == n_papers:
        return x.float()

    entry = store[ntype]
    for key in ("paper_idx", "paper_id", "paper", "doc_idx", "root_idx", "batch"):
        idx = None
        if hasattr(entry, key):
            idx = getattr(entry, key)
        elif isinstance(entry, dict) and key in entry:
            idx = entry[key]
        if isinstance(idx, torch.Tensor) and idx.numel() == x.shape[0]:
            out = torch.zeros(n_papers, x.shape[1], dtype=torch.float)
            cnt = torch.zeros(n_papers, 1, dtype=torch.float)
            out.index_add_(0, idx.long(), x.float())
            cnt.index_add_(0, idx.long(), torch.ones(x.shape[0], 1))
            return out / cnt.clamp(min=1.0)

    if hasattr(graph, "edge_types"):
        for et in graph.edge_types:
            src, _, dst = et
            if str(dst) == str(ntype) and str(src).lower() in PAPER_NODE_ALIASES:
                ei = graph[et].edge_index
                out = torch.zeros(n_papers, x.shape[1], dtype=torch.float)
                cnt = torch.zeros(n_papers, 1, dtype=torch.float)
                out.index_add_(0, ei[0].long(), x[ei[1].long()].float())
                cnt.index_add_(0, ei[0].long(), torch.ones(ei.shape[1], 1))
                return out / cnt.clamp(min=1.0)

    raise ValueError(
        f"node type '{ntype}' has {x.shape[0]} rows but the corpus has "
        f"{n_papers} papers, and no paper-index attribute or paper->{ntype} "
        f"edge type was found. Available attrs: "
        f"{[k for k in dir(entry) if not k.startswith('_')][:20]}")


def hetero_features(spec, graph, corpus) -> torch.Tensor:
    """(n, A, d) stacked hetero node features, one row per paper per aspect."""
    aspects = list(spec.aspects)
    store = _node_store(graph)
    mapping = resolve_node_map(graph, aspects)
    n = len(corpus["ids"])
    mats = []
    for asp in aspects:
        nt = mapping[asp]
        x = _get_x(store[nt])
        if x is None:
            raise ValueError(f"node type '{nt}' carries no feature matrix")
        mats.append(_align_to_papers(x, nt, graph, store, n))
    d = min(m.shape[1] for m in mats)
    if len({m.shape[1] for m in mats}) > 1:
        print(f"  [hetero] aspect feature dims differ "
              f"{[m.shape[1] for m in mats]}; truncating to {d}", flush=True)
    X = torch.stack([m[:, :d] for m in mats], dim=1)
    print(f"  [hetero] built {tuple(X.shape)} via {mapping}", flush=True)
    return X


def feature_bank(spec, corpus, graph, cache_dir: str,
                 source: str) -> Tuple[torch.Tensor, str]:
    """Return (X, actual_source). 'hetero' NEVER silently falls back."""
    if source == "hetero":
        X = hetero_features(spec, graph, corpus)            # raises on failure
        return X, "hetero"
    feats = adapters.aspect_features(spec, corpus, graph, cache_dir)
    got = feats["source"]
    if got == "hetero":
        # adapters found hetero when we asked for text: keep the request honest
        print("  [feat] adapters returned hetero for a 'text' request; "
              "recording actual source", flush=True)
    return feats["X"], got


# ==================================================================== training
def train_cell(X: torch.Tensor, cfg: Dict[str, Any],
               trainer: str) -> Tuple[Dict[str, Any], str]:
    """Train, then VERIFY which trainer actually ran.

    trainer: 'real'     -> requires GJEPA_TRAIN_HOOK to be set and to be used
             'fallback' -> requires the built-in diag.refmodel
    """
    prev = os.environ.get("GJEPA_TRAIN_HOOK")
    if trainer == "fallback":
        os.environ.pop("GJEPA_TRAIN_HOOK", None)
    elif trainer == "real":
        if not prev:
            raise RuntimeError(
                "cell requested the real trainer but GJEPA_TRAIN_HOOK is unset. "
                "export GJEPA_TRAIN_HOOK='train.paper_reason_gjepa:train_for_diag'")
    try:
        run = adapters.train_jepa(X, cfg)
    finally:
        if prev is None:
            os.environ.pop("GJEPA_TRAIN_HOOK", None)
        else:
            os.environ["GJEPA_TRAIN_HOOK"] = prev

    src = str(run.get("source", "unknown"))
    looks_fallback = ("refmodel" in src.lower() or "fallback" in src.lower()
                      or src == "unknown")
    actual = "fallback" if looks_fallback else "real"
    return run, actual


def _bank_ranks(run: Dict[str, Any], qidx: np.ndarray) -> torch.Tensor:
    q = torch.as_tensor(qidx, dtype=torch.long)
    r = ranks_from_banks(run["Q"][q].to(device()), run["C"].to(device()),
                         q.to(device()))
    return r.detach().to("cpu", dtype=torch.float64)


def _metrics(r: torch.Tensor, n: int) -> Dict[str, float]:
    b = bits_recovered(r, n)
    return {"mrr": mrr(r), "r@1": recall_at(r, 1),
            "bits": b["bits_recovered"], "bits_total": b["bits_total"]}


# ==================================================================== W2 probe
def ridge_probe(X: torch.Tensor, qidx: np.ndarray, query_aspect: int = 0,
                lam: float = 1e-2, train_frac: float = 0.5,
                seed: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Closed-form ridge from the visible-aspect mean to the masked aspect.

    Fitted on a disjoint split so the probe cannot memorise the eval queries.
    Returns (ranks over the FULL bank for qidx, metrics).
    """
    n, A, _ = X.shape
    Z = F.normalize(X.float(), dim=-1)
    vis = [a for a in range(A) if a != query_aspect]
    Ctx = F.normalize(Z[:, vis, :].mean(1), dim=-1)
    Tgt = Z[:, query_aspect, :]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    eval_set = set(int(i) for i in qidx)
    tr = np.array([i for i in perm if int(i) not in eval_set], dtype=np.int64)
    if len(tr) < 100:
        tr = perm[: max(int(train_frac * n), 100)]

    dev = device()
    Ctr = Ctx[torch.as_tensor(tr)].to(dev)
    Ttr = Tgt[torch.as_tensor(tr)].to(dev)
    d = Ctr.shape[1]
    G = Ctr.T @ Ctr + lam * len(tr) * torch.eye(d, device=dev)
    W = torch.linalg.solve(G, Ctr.T @ Ttr)

    q = torch.as_tensor(qidx, dtype=torch.long)
    Qh = F.normalize(Ctx[q].to(dev) @ W, dim=-1)
    r = ranks_from_banks(Qh, Tgt.to(dev), q.to(dev))
    r = r.detach().to("cpu", dtype=torch.float64)
    return r, {**_metrics(r, n), "n_train": int(len(tr)), "lambda": lam}


# ============================================================ W1 intervention
def rho_intervention(X: torch.Tensor, qidx: np.ndarray, cfg: Dict[str, Any],
                     trainer: str, doses: Sequence[float],
                     query_aspect: int = 0) -> List[Dict[str, float]]:
    """Dose-response curve: scale the aspect-mean, measure rho, measure bits.

    X_alpha[:, a, :] = normalize( alpha * mu_a  +  (Z[:, a, :] - mu_a) )

    Only the between-aspect component is scaled; the per-paper residual is
    untouched. So this is an intervention on rho in the REAL feature space,
    and the achieved rho is MEASURED rather than assumed.
    """
    n = X.shape[0]
    Z = F.normalize(X.float(), dim=-1)
    mu = Z.mean(0, keepdim=True)                    # (1, A, d) per-aspect mean
    R = Z - mu
    out = []
    for alpha in doses:
        Xa = F.normalize(alpha * mu + R, dim=-1)
        geo = _geometry(Xa)
        run, actual = train_cell(Xa, cfg, trainer)
        r = _bank_ranks(run, qidx)
        m = _metrics(r, n)
        # training-free reference in the same intervened space
        vis = [a for a in range(Xa.shape[1]) if a != query_aspect]
        Qo = F.normalize(F.normalize(Xa, dim=-1)[:, vis, :].mean(1), dim=-1)
        Co = F.normalize(Xa, dim=-1)[:, query_aspect, :]
        q = torch.as_tensor(qidx, dtype=torch.long)
        ro = ranks_from_banks(Qo[q].to(device()), Co.to(device()),
                              q.to(device())).detach().cpu().double()
        mo = _metrics(ro, n)
        row = {"alpha": float(alpha),
               "rho_between_aspect": geo["rho_between_aspect"],
               "rho_between_paper": geo["rho_between_paper"],
               "kappa": geo["kappa"],
               "trained_bits": m["bits"], "trained_mrr": m["mrr"],
               "oracle_bits": mo["bits"], "oracle_mrr": mo["mrr"],
               "trainer_actual": actual}
        out.append(row)
        print(f"    [W1:dose] alpha={alpha:>7.2f}  rho_asp={row['rho_between_aspect']:.5f}"
              f"  trained={m['bits']:+.3f}b  oracle={mo['bits']:+.3f}b", flush=True)
    return out


def _rho_crit(curve: List[Dict[str, float]], thresh: float) -> Optional[float]:
    """Smallest measured rho whose trained bits fall below `thresh`."""
    hits = [c["rho_between_aspect"] for c in curve if c["trained_bits"] < thresh]
    return float(min(hits)) if hits else None


# ======================================================================= tests
def test_w1(X, qidx, cfg, trainer, base_run, doses) -> Dict[str, Any]:
    """W1: does a high between-aspect share CAUSE the collapse?

    Splits the claim in two, because they need different evidence:
      (a) rho drives collapse   -> intervention dose-response curve
      (b) anisotropy is the residual factor -> sensitivity of bits to kappa
    """
    n = X.shape[0]
    geo = _geometry(X)
    r = _bank_ranks(base_run, qidx)
    obs = _metrics(r, n)
    collapsed = obs["bits"] < THRESHOLDS["collapse_bits"]
    print(f"  [W1] observed: rho_asp={geo['rho_between_aspect']:.5f} "
          f"rho_paper={geo['rho_between_paper']:.4f} kappa={geo['kappa']:.3f} "
          f"-> trained {obs['bits']:+.3f} bits "
          f"({'COLLAPSED' if collapsed else 'NO COLLAPSE'})", flush=True)

    curve = rho_intervention(X, qidx, cfg, trainer, doses)
    rho_c = _rho_crit(curve, THRESHOLDS["collapse_bits"])

    # (b) anisotropy: whiten the candidate spectrum, keep rho fixed, retrain
    Zc = F.normalize(X.float(), dim=-1)
    flat = Zc.reshape(-1, Zc.shape[-1])
    mu_f = flat.mean(0, keepdim=True)
    cov = ((flat - mu_f).T @ (flat - mu_f)) / max(len(flat) - 1, 1)
    ev, V = torch.linalg.eigh(cov.double())
    Wz = (V @ torch.diag(ev.clamp(min=1e-8) ** -0.5) @ V.T).float()
    Xw = F.normalize((Zc.reshape(-1, Zc.shape[-1]) - mu_f) @ Wz,
                     dim=-1).reshape(Zc.shape)
    geo_w = _geometry(Xw)
    run_w, _ = train_cell(Xw, cfg, trainer)
    bits_w = _metrics(_bank_ranks(run_w, qidx), n)["bits"]
    kappa_effect = abs(bits_w - obs["bits"])
    kappa_material = kappa_effect > THRESHOLDS["kappa_material_bits"]
    print(f"  [W1] whitened (kappa {geo['kappa']:.3f} -> {geo_w['kappa']:.3f}): "
          f"{bits_w:+.3f} bits, |delta|={kappa_effect:.3f} "
          f"-> anisotropy {'MATERIAL' if kappa_material else 'NOT material'}",
          flush=True)

    if collapsed and rho_c is not None and geo["rho_between_aspect"] >= rho_c:
        verdict = ("W1 SUPPORTED: the collapse is present (%.3f bits) and the "
                   "intervention shows bits fall below %.1f only for rho >= "
                   "%.5f, which the measured rho (%.5f) exceeds."
                   % (obs["bits"], THRESHOLDS["collapse_bits"], rho_c,
                      geo["rho_between_aspect"]))
        status = "supported"
    elif not collapsed:
        verdict = ("W1 PREMISE ABSENT: no collapse in this cell (%.3f of %.2f "
                   "bits). The intervention still locates a causal threshold at "
                   "rho >= %s, so the mechanism is real but the corpus does not "
                   "sit in that regime."
                   % (obs["bits"], obs["bits_total"],
                      "%.5f" % rho_c if rho_c else "none found"))
        status = "premise_absent"
    else:
        verdict = ("W1 REFUTED AS STATED: the collapse is present (%.3f bits) "
                   "but the intervention does not reproduce it from rho "
                   "(threshold %s vs measured %.5f). Another factor is "
                   "responsible."
                   % (obs["bits"], "%.5f" % rho_c if rho_c else "none found",
                      geo["rho_between_aspect"]))
        status = "refuted"
    print(f"  [W1] {verdict}", flush=True)
    return {"status": status, "geometry": geo, "observed": obs,
            "collapsed": collapsed, "intervention": curve,
            "rho_critical_measured": rho_c,
            "whitened": {"geometry": geo_w, "bits": bits_w,
                         "delta_bits": kappa_effect,
                         "anisotropy_material": kappa_material},
            "verdict": verdict}


def test_w2(X, qidx, base_run, seed=0) -> Dict[str, Any]:
    """W2: context linearly encodes identity, but the objective ignores it.

    Requires BOTH: probe high AND trained low, with a PAIRED test on the gap.
    """
    n = X.shape[0]
    r_tr = _bank_ranks(base_run, qidx)
    m_tr = _metrics(r_tr, n)
    r_pr, m_pr = ridge_probe(X, qidx, seed=seed)

    Z = F.normalize(X.float(), dim=-1)
    vis = list(range(1, Z.shape[1]))
    Qo = F.normalize(Z[:, vis, :].mean(1), dim=-1)
    q = torch.as_tensor(qidx, dtype=torch.long)
    r_or = ranks_from_banks(Qo[q].to(device()), Z[:, 0, :].to(device()),
                            q.to(device())).detach().cpu().double()
    m_or = _metrics(r_or, n)

    pb_probe = paired_bootstrap((1.0 / r_pr).numpy(), (1.0 / r_tr).numpy(),
                                alpha=THRESHOLDS["paired_alpha"], seed=seed)
    pb_oracle = paired_bootstrap((1.0 / r_or).numpy(), (1.0 / r_tr).numpy(),
                                 alpha=THRESHOLDS["paired_alpha"], seed=seed)

    print(f"  [W2] trained {m_tr['bits']:+.3f}b | ridge probe "
          f"{m_pr['bits']:+.3f}b | oracle {m_or['bits']:+.3f}b", flush=True)
    print(f"  [W2] paired dRR(probe-trained)={pb_probe['mean_diff']:+.4f} "
          f"[{pb_probe['lo']:+.4f},{pb_probe['hi']:+.4f}] p={pb_probe['p']:.4g}",
          flush=True)
    print(f"  [W2] paired dRR(oracle-trained)={pb_oracle['mean_diff']:+.4f} "
          f"[{pb_oracle['lo']:+.4f},{pb_oracle['hi']:+.4f}] "
          f"p={pb_oracle['p']:.4g}", flush=True)

    encodes = m_pr["bits"] > THRESHOLDS["probe_encodes_bits"]
    ignores = m_tr["bits"] < THRESHOLDS["collapse_bits"]
    gap_real = pb_probe["excludes_zero"] and pb_probe["mean_diff"] > 0
    established = bool(encodes and ignores and gap_real)

    if established:
        verdict = ("W2 SUPPORTED: the probe recovers %.3f bits while the "
                   "objective recovers %.3f, and the paired gap excludes zero "
                   "(p=%.3g)." % (m_pr["bits"], m_tr["bits"], pb_probe["p"]))
    elif not ignores:
        verdict = ("W2 NOT SUPPORTED: the objective does NOT ignore the signal "
                   "(%.3f bits, above the %.1f-bit collapse threshold). The "
                   "paired probe-minus-trained difference is %+.4f RR "
                   "[%+.4f,%+.4f], so the dissociation as written does not "
                   "hold in this cell."
                   % (m_tr["bits"], THRESHOLDS["collapse_bits"],
                      pb_probe["mean_diff"], pb_probe["lo"], pb_probe["hi"]))
    else:
        verdict = ("W2 INCONCLUSIVE: the objective collapses (%.3f bits) but "
                   "the probe does not clear the %.1f-bit encoding threshold "
                   "(%.3f), so there is no demonstrated signal for it to "
                   "ignore." % (m_tr["bits"], THRESHOLDS["probe_encodes_bits"],
                                m_pr["bits"]))
    print(f"  [W2] {verdict}", flush=True)
    return {"status": "supported" if established else "not_supported",
            "trained": m_tr, "ridge_probe": m_pr, "oracle": m_or,
            "paired_probe_vs_trained": pb_probe,
            "paired_oracle_vs_trained": pb_oracle,
            "context_encodes": encodes, "objective_ignores": ignores,
            "verdict": verdict}


def test_w6(X, qidx, cfg, trainer, base_run, seed=0) -> Dict[str, Any]:
    """W6: can the pipeline learn retrievability at all?

    Degeneracy guard: the winning control must differ from the model under test
    in CONFIG and by more than degenerate_bits_eps in VALUE. The previous run
    reported a control identical to the model under test to four decimals.
    """
    n = X.shape[0]
    mut_bits = _metrics(_bank_ranks(base_run, qidx), n)["bits"]
    r_mut = _bank_ranks(base_run, qidx)

    def same_cfg(a, b):
        ka = {k: v for k, v in a.items() if k != "name"}
        kb = {k: v for k, v in b.items() if k != "name"}
        return ka == kb

    results, degenerate = {}, []
    for spec in CONTROL_SPECS:
        if same_cfg(spec, MODEL_UNDER_TEST):
            degenerate.append(spec["name"])
            print(f"  [W6] EXCLUDED '{spec['name']}': config identical to the "
                  f"model under test", flush=True)
            continue
        c = {**cfg, **{k: v for k, v in spec.items() if k != "name"},
             "seed": cfg.get("seed", seed)}
        try:
            run, actual = train_cell(X, c, trainer)
        except Exception as e:                                   # noqa: BLE001
            print(f"  [W6] control '{spec['name']}' FAILED: {e}", flush=True)
            results[spec["name"]] = {"status": "error", "err": str(e)[:200]}
            continue
        r = _bank_ranks(run, qidx)
        m = _metrics(r, n)
        pb = paired_bootstrap((1.0 / r).numpy(), (1.0 / r_mut).numpy(),
                              alpha=THRESHOLDS["paired_alpha"], seed=seed)
        num_deg = abs(m["bits"] - mut_bits) < THRESHOLDS["degenerate_bits_eps"]
        if num_deg:
            degenerate.append(spec["name"])
        results[spec["name"]] = {"status": "ok", **m, "trainer_actual": actual,
                                 "paired_vs_mut": pb,
                                 "numerically_degenerate": num_deg}
        print(f"  [W6] {spec['name']:20s} {m['bits']:+.3f}b  "
              f"dRR vs MUT {pb['mean_diff']:+.4f} "
              f"[{pb['lo']:+.4f},{pb['hi']:+.4f}] p={pb['p']:.3g}"
              f"{'  <-- DEGENERATE' if num_deg else ''}", flush=True)

    eligible = {k: v for k, v in results.items()
                if v.get("status") == "ok" and not v["numerically_degenerate"]}
    if not eligible:
        verdict = ("W6 VACUOUS: no eligible control survived. Excluded/"
                   "degenerate: %s. A control identical to the model under test "
                   "carries no information about pipeline capability."
                   % (degenerate or "none"))
        status = "vacuous"
        best = None
    else:
        best = max(eligible, key=lambda k: eligible[k]["bits"])
        bb = eligible[best]
        capable = bb["bits"] > THRESHOLDS["control_capable_bits"]
        beats = bb["paired_vs_mut"]["excludes_zero"] and \
            bb["paired_vs_mut"]["mean_diff"] > 0
        if capable and beats:
            verdict = ("W6 RULED OUT (pipeline is capable): control '%s' "
                       "recovers %.3f bits versus %.3f for the model under "
                       "test, paired difference %+.4f RR [%+.4f,%+.4f], "
                       "p=%.3g. Failure is attributable to the objective, not "
                       "the pipeline."
                       % (best, bb["bits"], mut_bits,
                          bb["paired_vs_mut"]["mean_diff"],
                          bb["paired_vs_mut"]["lo"], bb["paired_vs_mut"]["hi"],
                          bb["paired_vs_mut"]["p"]))
            status = "ruled_out"
        elif capable:
            verdict = ("W6 NOT DECISIVE: the best control '%s' is capable "
                       "(%.3f bits) but does not significantly beat the model "
                       "under test (%.3f bits, paired p=%.3g). There is no "
                       "objective-specific failure to attribute."
                       % (best, bb["bits"], mut_bits,
                          bb["paired_vs_mut"]["p"]))
            status = "not_decisive"
        else:
            verdict = ("W6 CONFIRMS A PIPELINE PROBLEM: even the best control "
                       "'%s' recovers only %.3f bits, below the %.1f-bit "
                       "capability threshold. The failure is NOT specific to "
                       "the objective."
                       % (best, bb["bits"], THRESHOLDS["control_capable_bits"]))
            status = "pipeline_problem"
    print(f"  [W6] {verdict}", flush=True)
    return {"status": status, "model_under_test": MODEL_UNDER_TEST,
            "mut_bits": mut_bits, "controls": results,
            "excluded_or_degenerate": degenerate, "best_eligible": best,
            "verdict": verdict}


# ======================================================================= cells
def run_cell(spec, corpus, graph, cache_dir: str, trainer: str, feats: str,
             args) -> Dict[str, Any]:
    tag = f"{trainer}+{feats}"
    print(f"\n{'=' * 70}\n  CELL {tag}\n{'=' * 70}", flush=True)
    t0 = time.time()
    out: Dict[str, Any] = {"cell": tag, "requested": {"trainer": trainer,
                                                      "features": feats}}
    try:
        X, actual_feats = feature_bank(spec, corpus, graph, cache_dir, feats)
    except Exception as e:                                       # noqa: BLE001
        print(f"  [cell {tag}] feature build FAILED: {e}", flush=True)
        out.update({"status": "provenance_mismatch",
                    "reason": f"features unavailable: {e}",
                    "seconds": time.time() - t0})
        return out

    feats_ok = (actual_feats == "hetero") if feats == "hetero" \
        else (actual_feats != "hetero")
    if not feats_ok:
        msg = (f"asked for '{feats}' features, adapters supplied "
               f"'{actual_feats}'. No verdict emitted.")
        print(f"  [cell {tag}] PROVENANCE MISMATCH: {msg}", flush=True)
        out.update({"status": "provenance_mismatch", "reason": msg,
                    "actual_features": actual_feats,
                    "seconds": time.time() - t0})
        return out

    n = X.shape[0]
    nq = int(min(args.n_queries or n, n))
    rng = np.random.default_rng(args.seed)
    qidx = np.sort(rng.choice(n, size=nq, replace=False)) if nq < n else np.arange(n)
    print(f"  [cell {tag}] X={tuple(X.shape)} source={actual_feats}  "
          f"queries={nq}/{n}", flush=True)

    cfg = {"epochs": args.epochs, "seed": args.seed,
           **{k: v for k, v in MODEL_UNDER_TEST.items() if k != "name"}}
    try:
        base_run, actual_trainer = train_cell(X, cfg, trainer)
    except Exception as e:                                       # noqa: BLE001
        print(f"  [cell {tag}] training FAILED: {e}", flush=True)
        out.update({"status": "provenance_mismatch",
                    "reason": f"trainer unavailable: {e}",
                    "actual_features": actual_feats,
                    "seconds": time.time() - t0})
        return out

    if actual_trainer != trainer:
        msg = (f"asked for the '{trainer}' trainer, got '{actual_trainer}' "
               f"(run source='{base_run.get('source')}'). No verdict emitted.")
        print(f"  [cell {tag}] PROVENANCE MISMATCH: {msg}", flush=True)
        out.update({"status": "provenance_mismatch", "reason": msg,
                    "actual_features": actual_feats,
                    "actual_trainer": actual_trainer,
                    "seconds": time.time() - t0})
        return out

    out.update({"status": "ok", "actual_features": actual_feats,
                "actual_trainer": actual_trainer,
                "n": n, "n_queries": nq, "epochs": args.epochs})
    doses = [float(x) for x in args.doses.split(",") if x.strip()]
    icfg = {**cfg, "epochs": args.intervention_epochs or args.epochs}
    out["W1"] = test_w1(X, qidx, icfg, trainer, base_run, doses)
    out["W2"] = test_w2(X, qidx, base_run, seed=args.seed)
    out["W6"] = test_w6(X, qidx, cfg, trainer, base_run, seed=args.seed)
    out["seconds"] = time.time() - t0
    return out


# ======================================================================== main
def _cross_cell(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Attribute any disagreement to features, to the trainer, or to neither."""
    ok = {c["cell"]: c for c in cells if c.get("status") == "ok"}
    note = []
    def bits(tag):
        c = ok.get(tag)
        return None if c is None else c["W2"]["trained"]["bits"]
    for tr in ("fallback", "real"):
        a, b = bits(f"{tr}+text"), bits(f"{tr}+hetero")
        if a is not None and b is not None:
            note.append(f"features effect at trainer={tr}: "
                        f"text {a:+.3f}b vs hetero {b:+.3f}b (delta {b - a:+.3f})")
    for ft in ("text", "hetero"):
        a, b = bits(f"fallback+{ft}"), bits(f"real+{ft}")
        if a is not None and b is not None:
            note.append(f"trainer effect at features={ft}: "
                        f"fallback {a:+.3f}b vs real {b:+.3f}b (delta {b - a:+.3f})")
    if not note:
        note.append("no two cells share a factor; nothing is attributable yet")
    for s in note:
        print(f"  [attrib] {s}", flush=True)
    return {"notes": note, "cells_ok": sorted(ok),
            "cells_blocked": sorted(c["cell"] for c in cells
                                    if c.get("status") != "ok")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("diag.stages_w126")
    ap.add_argument("--corpus", default="papers")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cells",
                    default="fallback+text,fallback+hetero,real+text,real+hetero")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--intervention-epochs", type=int, default=40,
                    help="shorter training for the dose-response sweep")
    ap.add_argument("--doses", default="0,0.5,1,3,10,30,100",
                    help="aspect-mean scale factors for the W1 intervention")
    ap.add_argument("--n-queries", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-papers", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=20)
    a = ap.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    prereg = os.path.join(a.out_dir, "w126_prereg.json")
    with open(prereg, "w") as f:
        json.dump({"thresholds": THRESHOLDS,
                   "model_under_test": MODEL_UNDER_TEST,
                   "controls": CONTROL_SPECS,
                   "argv": " ".join(sys.argv[1:]),
                   "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"  [prereg] thresholds fixed and written to {prereg}", flush=True)
    for k, v in THRESHOLDS.items():
        print(f"    {k:26s} {v}", flush=True)

    set_seed(a.seed)
    spec = adapters.get_spec(a.corpus)
    corpus = adapters.load_texts(spec, limit=a.limit_papers or None,
                                 min_chars=a.min_chars)
    graph = adapters.load_graph(spec)
    cache_dir = os.path.join(a.out_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        store = _node_store(graph)
        print(f"  [graph] node types: {list(store.keys())}", flush=True)
        print(f"  [graph] resolved map: "
              f"{resolve_node_map(graph, list(spec.aspects))}", flush=True)
    except Exception as e:                                       # noqa: BLE001
        print(f"  [graph] node-type introspection failed: {e}", flush=True)

    cells: List[Dict[str, Any]] = []
    for tag in [t.strip() for t in a.cells.split(",") if t.strip()]:
        trainer, _, feats = tag.partition("+")
        if trainer not in ("fallback", "real") or feats not in ("text", "hetero"):
            print(f"  [skip] malformed cell '{tag}'", flush=True)
            continue
        try:
            cells.append(run_cell(spec, corpus, graph, cache_dir,
                                  trainer, feats, a))
        except Exception as e:                                   # noqa: BLE001
            traceback.print_exc()
            cells.append({"cell": tag, "status": "error", "err": str(e)[:300]})

    print(f"\n{'=' * 70}\n  CROSS-CELL ATTRIBUTION\n{'=' * 70}", flush=True)
    attrib = _cross_cell(cells)

    print(f"\n{'-' * 70}\n  W1/W2/W6 SUMMARY\n{'-' * 70}")
    print(f"  {'cell':22s} {'W1':16s} {'W2':16s} {'W6':16s}")
    for c in cells:
        if c.get("status") != "ok":
            print(f"  {c['cell']:22s} {c.get('status', '?'):16s} "
                  f"{'-':16s} {'-':16s}  {c.get('reason', c.get('err', ''))[:60]}")
        else:
            print(f"  {c['cell']:22s} {c['W1']['status']:16s} "
                  f"{c['W2']['status']:16s} {c['W6']['status']:16s}")

    path = os.path.join(a.out_dir, f"w126_{a.corpus}.json")
    with open(path, "w") as f:
        json.dump({"thresholds": THRESHOLDS, "cells": cells,
                   "attribution": attrib}, f, indent=2, default=str)
    print(f"\n  results: {path}")
    blocked = [c for c in cells if c.get("status") != "ok"]
    return 0 if not blocked else 2


if __name__ == "__main__":
    sys.exit(main())