"""
paper_reason_euclidean_fix.py — GAP 1 + GAP 2 (reviewer-requested experiments)
                                + full TIMING / GPU-HOUR / COST instrumentation.
============================================================================
Self-contained. Imports ONLY from the two existing training files; modifies
nothing in them. Add this file to train/ and run:

    python -m train.paper_reason_euclidean_fix

============================================================================
WHAT THIS FILE ADDS (and why)
============================================================================
GAP 1 — TRAINED EUCLIDEAN-COSINE JEPA (the paper's "recommended fix", run
        end-to-end instead of only as a training-free oracle).
  * train_euclidean(): identical JEPA dynamics to train_multi (EMA target,
    stop-grad, RWSE PE, latent->latent predictor) EXCEPT the target geometry.
    Loss = 1 - cos(pred, target) on the RAW latents. No hyperbola, no Lorentz,
    no scalar mean-angle -> no rank-2 target bottleneck.
  * eval_patchret_cosine(): same query construction as eval_patchret, but ranks
    candidates by COSINE similarity of L2-normalized latents. Column 0 is the
    gold; it is NOT fed to the query path, so there is no self-match leak.
  * Sweep {raw, whitened} x {mean, relhetero} over the 5 seeds, printed next to
    the collapsed hyperbolic row (Table 2) and the oracle ceiling (Table 6).
  * Extras: full-58k-pool toggle and a claim-only retrieval control.

GAP 2 — ISOLATED PRE-ANCHOR POOLED-LATENT RANK (Reviewer 2's exact ask).
  * run_gap2_preanchor(): trains a plain mean-pool encoder with the SAME JEPA/
    EMA dynamics but NO anchors and NO hyperbolic target, then reports pm_rk.

============================================================================
TIMING / COST INSTRUMENTATION (this version)
============================================================================
  * Timer context manager: wall-clock for every phase (build, whiten, GAP2,
    each GAP1 cell, oracle) with peak GPU memory (torch.cuda.max_memory_allocated).
  * Per-seed train + eval seconds logged inside every cell.
  * GPU-hour accounting: total_gpu_hours = wall_hours * NUM_GPUS_BILLED.
  * COST estimate: GPU_HOURLY_RATE_USD x total_gpu_hours (rate is configurable;
    default reflects a common L40S on-demand price — EDIT to your provider).
  * A COST_SUMMARY block + a machine-readable "cost" section in both JSONs.
  * NOTE: numbers are MEASURED at runtime on YOUR hardware. The dollar figure is
    only as accurate as GPU_HOURLY_RATE_USD and NUM_GPUS_BILLED, which you set.
============================================================================
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import time
import random
import platform
import contextlib
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from core.data_utils.paper_graph import build_hetero_graph, ASPECTS

# ---- reuse the ORIGINAL (frozen) core ----
import train.paper_reason_gjepa_old_1 as G
from train.paper_reason_gjepa_old_1 import (
    DEVICE, HIDDEN, LATENT, EPOCHS, LR, WD, SEEDS,
    EMA_BASE, EMA_FINAL,
    RAW_DIR, CACHE_PATH, RWSE_CACHE, CKPT_DIR,
    mlp3, collapse_stats, eff_rank_only,
    build_patch_index, aspect_presence, build_reasoning_label,
    eval_probe, mean_std, verdict,
)

# ---- reuse the v2-v6 additions (model, whitening, relhetero, oracle) ----
from train.paper_reason_gjepa_old_2 import (
    GraphJEPAMulti, whiten_features, build_hetero_neighbors,
    oracle_retrieval_test, REL_POOL, WHITEN_EPS, REL_MAX_NEIGH,
)

# ─────────────────────────────────────────────────────────────
EUC_RESULTS_JSON       = os.path.join(CKPT_DIR, "gjepa_euclidean_fix_results.json")
PREANCHOR_RESULTS_JSON = os.path.join(CKPT_DIR, "gjepa_preanchor_rank_results.json")

# Objective config for the Euclidean-cosine variant.
EUC_CFG = dict(loss_mode="euclidean", use_vic=False, inv=1.0, std=0.0, cov=0.0)

# Retrieval eval knobs
RET_NEG      = 1000
RET_QUERY_BS = 2048
FULL_POOL    = False

# ─────────────────────────────────────────────────────────────
#  COST / GPU-HOUR CONFIG  — EDIT THESE TWO TO MATCH YOUR PROVIDER
# ─────────────────────────────────────────────────────────────
NUM_GPUS_BILLED   = 1        # GPUs you are billed for during this job (you use 2x L40S)
GPU_HOURLY_RATE_USD = 1.00   # USD per GPU-hour (EDIT: e.g. L40S on-demand ~ $1.0-1.4/hr)
COST_CURRENCY     = "USD"
# ─────────────────────────────────────────────────────────────


# ============================================================
#  TIMING UTILITIES
# ============================================================
def _gpu_reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _gpu_peak_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


class Timer(contextlib.AbstractContextManager):
    """Wall-clock + peak-GPU-memory timer. Appends a record to `sink` (a list)."""
    def __init__(self, name, sink):
        self.name = name; self.sink = sink
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _gpu_reset_peak()
        self.t0 = time.time()
        return self
    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - self.t0
        peak = _gpu_peak_gb()
        self.sink.append({"phase": self.name, "seconds": dt, "peak_gpu_gb": peak})
        print(f"  [time] {self.name:<40s} {dt:8.1f}s  ({dt/60:5.1f} min)  "
              f"peak_gpu={peak:5.2f} GB", flush=True)
        return False


def _fmt_hms(sec):
    sec = int(round(sec)); h = sec // 3600; m = (sec % 3600) // 60; s = sec % 60
    return f"{h:d}h{m:02d}m{s:02d}s"


# ============================================================
#  GAP 1a — TRAIN a Euclidean-cosine JEPA (no rank-2 target)
#  Returns (..., timings) with train/eval seconds separated.
# ============================================================
def train_euclidean(seed, data, active, patch_idx, pres, rwse, maskable, pe_dim,
                    pool_mode="mean", rel_neighbors=None, whitened_x=None):
    torch.manual_seed(seed)
    full = data.to(DEVICE)
    P = full["paper"].num_nodes; A = len(active)
    pe = {a: rwse[a].to(DEVICE) for a in active}
    pres_d = pres.to(DEVICE); maskable_idx = maskable.to(DEVICE)
    model = GraphJEPAMulti(data.metadata(), HIDDEN, LATENT, active, pe_dim, pool_mode,
                           mah_k=0, rel_neighbors=rel_neighbors).to(DEVICE)

    x_in = full.x_dict
    if whitened_x is not None:
        x_in = {k: (whitened_x[k].to(DEVICE) if k in whitened_x else full.x_dict[k])
                for k in full.x_dict}

    with torch.no_grad():
        model.encode_nodes(x_in, full.edge_index_dict)
        model.encode_nodes_tgt(x_in, full.edge_index_dict)
    model.init_target(); model.freeze_target()

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=LR, weight_decay=WD)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_train0 = time.time()

    for ep in range(EPOCHS):
        model.train()
        m = EMA_FINAL - (EMA_FINAL - EMA_BASE) * (0.5 * (1 + np.cos(np.pi * ep / max(1, EPOCHS - 1))))
        with torch.no_grad():
            zt = model.encode_nodes_tgt(x_in, full.edge_index_dict)
        zc = model.encode_nodes(x_in, full.edge_index_dict)

        ctx_emb = torch.stack([model.patch_embed(zc, patch_idx, a, P, "ctx", node_pe=pe)
                               for a in active], 1)
        with torch.no_grad():
            zt_full = {k: zt[k].detach() for k in zt} if pool_mode == "relhetero" \
                      else {k: zt[k].detach() for k in active}
            tgt_emb = torch.stack([model.patch_embed(zt_full, patch_idx, a, P, "tgt", node_pe=pe)
                                   for a in active], 1)

        Pm = maskable_idx.size(0)
        pres_m = pres_d[maskable_idx]
        rnd = torch.rand(Pm, A, generator=gen, device=DEVICE); rnd[~pres_m] = -1.0
        tgt_aspect = rnd.argmax(1)

        ctx_m = ctx_emb[maskable_idx]; tgt_m = tgt_emb[maskable_idx]
        ctx_mask = pres_m.clone()
        ctx_mask[torch.arange(Pm, device=DEVICE), tgt_aspect] = False
        mixed = model.ctx_mixer(ctx_m, src_key_padding_mask=~ctx_mask)
        w = ctx_mask.float().unsqueeze(-1)
        ctx_summary = (mixed * w).sum(1) / w.sum(1).clamp_min(1.0)

        pe_stack = torch.stack([pe[a] for a in active], 1)
        tgt_pe = pe_stack[maskable_idx][torch.arange(Pm, device=DEVICE), tgt_aspect]

        pred_latent = model.predictor(ctx_summary + model.pe_proj(tgt_pe))
        tgt_patch   = tgt_m[torch.arange(Pm, device=DEVICE), tgt_aspect]

        pred_n = F.normalize(pred_latent, dim=-1)
        tgt_n  = F.normalize(tgt_patch,   dim=-1)
        inv_loss = (1.0 - (pred_n * tgt_n).sum(-1)).mean()
        loss = inv_loss

        opt.zero_grad(); loss.backward(); opt.step(); model.ema(m)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_sec = time.time() - t_train0

    model.eval()
    t_eval0 = time.time()
    with torch.no_grad():
        zt = model.encode_nodes_tgt(x_in, full.edge_index_dict)
        zt_full = {k: zt[k] for k in zt} if pool_mode == "relhetero" \
                  else {k: zt[k] for k in active}
        patch_repr = torch.stack(
            [model.patch_embed(zt_full, patch_idx, a, P, "tgt", node_pe=pe)
             for a in active], 1)
        node_ranks = {a: eff_rank_only(zt[a]) for a in active}
        pm_flat = patch_repr[maskable_idx][pres_d[maskable_idx]]
        vals = [v for v in node_ranks.values() if not np.isnan(v)]
        upstream = {"node_eff_rank_mean": float(np.mean(vals)) if vals else float("nan"),
                    "node_eff_rank_per_aspect": {a: float(node_ranks[a]) for a in active},
                    "patchmean_eff_rank": float(eff_rank_only(pm_flat))}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rankprobe_sec = time.time() - t_eval0
    timings = {"train_sec": train_sec, "rankprobe_sec": rankprobe_sec,
               "peak_gpu_gb": _gpu_peak_gb()}
    return model, patch_repr, pe, pres_d, maskable_idx, upstream, timings


# ============================================================
#  GAP 1b — RETRIEVAL by COSINE similarity (Euclidean space)
# ============================================================
@torch.no_grad()
def eval_patchret_cosine(model, patch_repr, pe, pres_d, active, maskable_idx, seed,
                         neg=RET_NEG, query_bs=RET_QUERY_BS, full_pool=FULL_POOL,
                         restrict_aspect=None):
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 777)
    P, A, L = patch_repr.shape
    pres_m = pres_d[maskable_idx]; Pm = maskable_idx.size(0)
    if Pm < 2:
        return {"MRR": float("nan"), "Hits@1": float("nan"),
                "Hits@10": float("nan"), **collapse_stats(patch_repr.reshape(-1, L))}

    if restrict_aspect is not None and restrict_aspect in active:
        ai = active.index(restrict_aspect)
        has_a = pres_m[:, ai]
        if int(has_a.sum()) < 2:
            return {"MRR": float("nan"), "Hits@1": float("nan"),
                    "Hits@10": float("nan"), **collapse_stats(patch_repr.reshape(-1, L))}
        sel = torch.nonzero(has_a, as_tuple=False).squeeze(1)
        maskable_idx = maskable_idx[sel]; pres_m = pres_m[sel]; Pm = maskable_idx.size(0)
        tgt_aspect = torch.full((Pm,), ai, device=DEVICE, dtype=torch.long)
    else:
        rnd = torch.rand(Pm, A, generator=gen, device=DEVICE); rnd[~pres_m] = -1.0
        tgt_aspect = rnd.argmax(1)

    ctx_m = patch_repr[maskable_idx]
    ctx_mask = pres_m.clone(); ctx_mask[torch.arange(Pm, device=DEVICE), tgt_aspect] = False
    mixed = model.ctx_mixer(ctx_m, src_key_padding_mask=~ctx_mask)
    w = ctx_mask.float().unsqueeze(-1)
    ctx_summary = (mixed * w).sum(1) / w.sum(1).clamp_min(1.0)
    pe_stack = torch.stack([pe[a] for a in active], 1)
    tgt_pe = pe_stack[maskable_idx][torch.arange(Pm, device=DEVICE), tgt_aspect]
    pred = model.predictor(ctx_summary + model.pe_proj(tgt_pe))
    pool = patch_repr[maskable_idx][torch.arange(Pm, device=DEVICE), tgt_aspect]
    cstats = collapse_stats(pool)

    pred_n = F.normalize(pred, dim=-1)
    pool_n = F.normalize(pool, dim=-1)

    ranks = torch.empty(Pm, device=DEVICE)
    for start in range(0, Pm, query_bs):
        end = min(start + query_bs, Pm); b = end - start
        true_local = torch.arange(start, end, device=DEVICE)
        q = pred_n[start:end]
        if full_pool or Pm <= neg + 1:
            sims = q @ pool_n.T
            gold_sim = sims[torch.arange(b, device=DEVICE), true_local].unsqueeze(1)
            ranks[start:end] = (sims > gold_sim).sum(1).float() + 1
        else:
            r = torch.randint(0, Pm, (b, neg), generator=gen, device=DEVICE)
            cand = torch.cat([true_local.view(-1, 1), r], 1)
            cp = pool_n[cand]
            sims = (q.unsqueeze(1) * cp).sum(-1)
            ranks[start:end] = (sims > sims[:, :1]).sum(1).float() + 1

    return {"MRR": float((1 / ranks).mean()),
            "Hits@1": float((ranks <= 1).float().mean()),
            "Hits@10": float((ranks <= 10).float().mean()), **cstats}


# ============================================================
#  GAP 1 DRIVER — sweep cell (with timing accumulation)
# ============================================================
def run_euclidean_cell(pool_mode, use_white, data, active, patch_idx, pres, label,
                       rwse, maskable, pe_dim, rel_neighbors, whitened_x,
                       full_pool=FULL_POOL, restrict_aspect=None):
    tag = f"euclidean | pool={pool_mode} | feats={'whitened' if use_white else 'raw'}"
    if restrict_aspect: tag += f" | aspect={restrict_aspect}"
    if full_pool: tag += " | FULL_POOL"
    print(f"\n### {tag}")
    A1, A2, UPS = [], [], []
    seed_train_sec, seed_eval_sec, seed_peak = [], [], []
    wx = whitened_x if use_white else None
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        model, patch_repr, pe, pres_d, maskable_idx, ups, tm = train_euclidean(
            s, data, active, patch_idx, pres, rwse, maskable, pe_dim,
            pool_mode=pool_mode, rel_neighbors=rel_neighbors, whitened_x=wx)
        t_ret0 = time.time()
        r1 = eval_probe(patch_repr, pres_d, active, label, s)
        r2 = eval_patchret_cosine(model, patch_repr, pe, pres_d, active, maskable_idx, s,
                                  full_pool=full_pool, restrict_aspect=restrict_aspect)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ret_sec = time.time() - t_ret0
        A1.append(r1); A2.append(r2); UPS.append(ups)
        seed_train_sec.append(tm["train_sec"])
        seed_eval_sec.append(tm["rankprobe_sec"] + ret_sec)
        seed_peak.append(tm["peak_gpu_gb"])
        print(f"  seed {s} | A1 {r1['acc']:.3f} | MRR {r2['MRR']:.3f} H@10 {r2['Hits@10']:.3f} "
              f"| pm_rk {ups['patchmean_eff_rank']:.1f} "
              f"| train {tm['train_sec']:.0f}s eval {tm['rankprobe_sec']+ret_sec:.0f}s "
              f"peak {tm['peak_gpu_gb']:.1f}GB", flush=True)
        del model, patch_repr, pe, pres_d, maskable_idx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    acc  = mean_std([r["acc"] for r in A1]); mrr = mean_std([r["MRR"] for r in A2])
    h10  = mean_std([r["Hits@10"] for r in A2]); h1 = mean_std([r["Hits@1"] for r in A2])
    lstd = mean_std([r["latent_std"] for r in A2]); erank = mean_std([r["eff_rank"] for r in A2])
    nrank= mean_std([u["node_eff_rank_mean"] for u in UPS])
    pmrank=mean_std([u["patchmean_eff_rank"] for u in UPS])
    cell_sec = float(np.sum(seed_train_sec) + np.sum(seed_eval_sec))
    return {"objective": "metis_euclidean", "pool_mode": pool_mode,
            "features": "whitened" if use_white else "raw",
            "restrict_aspect": restrict_aspect, "full_pool": full_pool,
            "A1_acc": acc, "MRR": mrr, "Hits@1": h1, "Hits@10": h10,
            "pool_eff_rank": erank, "node_eff_rank": nrank, "patchmean_eff_rank": pmrank,
            "verdict": verdict(lstd[0], erank[0]),
            "timing": {"cell_total_sec": cell_sec,
                       "train_sec_mean": float(np.mean(seed_train_sec)),
                       "eval_sec_mean": float(np.mean(seed_eval_sec)),
                       "peak_gpu_gb_max": float(np.max(seed_peak)),
                       "n_seeds": len(SEEDS)}}


# ============================================================
#  GAP 2 — isolated pre-anchor pooled-latent rank (with timing)
# ============================================================
def run_gap2_preanchor(data, active, patch_idx, pres, label, rwse, maskable, pe_dim):
    print("\n" + "#" * 100)
    print("#  GAP 2: isolated PRE-ANCHOR pooled-latent effective rank")
    print("#  (mean-pool JEPA, Euclidean-cosine target, NO anchors) — R2's de-confound")
    print("#" * 100)
    PMS = []; train_secs = []; peaks = []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        _, patch_repr, _, pres_d, maskable_idx, ups, tm = train_euclidean(
            s, data, active, patch_idx, pres, rwse, maskable, pe_dim, pool_mode="mean")
        PMS.append(ups); train_secs.append(tm["train_sec"]); peaks.append(tm["peak_gpu_gb"])
        print(f"  seed {s} | node_rk {ups['node_eff_rank_mean']:.1f} "
              f"pm_rk {ups['patchmean_eff_rank']:.1f} | train {tm['train_sec']:.0f}s "
              f"peak {tm['peak_gpu_gb']:.1f}GB", flush=True)
        del patch_repr, pres_d, maskable_idx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    nrank  = mean_std([u["node_eff_rank_mean"] for u in PMS])
    pmrank = mean_std([u["patchmean_eff_rank"] for u in PMS])
    print(f"\n  PRE-ANCHOR pooled-latent eff_rank (pm_rk) = {pmrank[0]:.2f} ± {pmrank[1]:.2f}")
    print(f"  (node-latent eff_rank = {nrank[0]:.1f}). If pm_rk ~ 2, the anchors receive")
    print(f"  near-1D latents => the collapse is UPSTREAM of the target geometry.")
    return {"node_eff_rank": nrank, "patchmean_eff_rank_preanchor": pmrank, "seeds": SEEDS,
            "timing": {"total_sec": float(np.sum(train_secs)),
                       "train_sec_mean": float(np.mean(train_secs)),
                       "peak_gpu_gb_max": float(np.max(peaks)), "n_seeds": len(SEEDS)}}


# ============================================================
#  COST / GPU-HOUR SUMMARY
# ============================================================
def build_cost_summary(phase_records, gap2, gap1_cells, oracle_sec, wall_sec):
    """Aggregate MEASURED wall-clock into GPU-hours and a $ estimate."""
    wall_hours = wall_sec / 3600.0
    gpu_hours = wall_hours * NUM_GPUS_BILLED
    cost = gpu_hours * GPU_HOURLY_RATE_USD

    gap1_train = sum(c["timing"]["cell_total_sec"] for c in gap1_cells)
    gap2_sec   = gap2["timing"]["total_sec"]
    peak_gb = max([p["peak_gpu_gb"] for p in phase_records] + [0.0])

    return {
        "hardware": {
            "device": DEVICE,
            "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            "num_gpus_billed": NUM_GPUS_BILLED,
            "platform": platform.platform(),
        },
        "rate": {"gpu_hourly_rate": GPU_HOURLY_RATE_USD, "currency": COST_CURRENCY},
        "config": {"epochs": EPOCHS, "seeds": len(SEEDS), "latent": LATENT,
                   "full_pool": FULL_POOL, "ret_neg": RET_NEG},
        "measured_seconds": {
            "wall_total": wall_sec,
            "gap2_preanchor": gap2_sec,
            "gap1_all_cells": gap1_train,
            "oracle": oracle_sec,
            "phase_breakdown": phase_records,
        },
        "gpu_hours": {
            "wall_hours": wall_hours,
            "gpu_hours_billed": gpu_hours,
        },
        "peak_gpu_gb": peak_gb,
        "estimated_cost": {
            "amount": round(cost, 4),
            "currency": COST_CURRENCY,
            "formula": "wall_hours * num_gpus_billed * gpu_hourly_rate",
            "note": "Estimate only; accuracy depends on GPU_HOURLY_RATE_USD and "
                    "NUM_GPUS_BILLED which you set at the top of this file.",
        },
    }


def print_cost_summary(cs):
    print("\n" + "=" * 90)
    print("  COST / GPU-HOUR SUMMARY  (measured on this run)")
    print("=" * 90)
    hw = cs["hardware"]
    print(f"  GPU                : {hw['gpu_name']}  x{hw['num_gpus_billed']} billed")
    print(f"  Epochs x seeds     : {cs['config']['epochs']} x {cs['config']['seeds']}")
    print(f"  Peak GPU memory    : {cs['peak_gpu_gb']:.2f} GB")
    ms = cs["measured_seconds"]
    print(f"  GAP 2 (pre-anchor) : {_fmt_hms(ms['gap2_preanchor'])}")
    print(f"  GAP 1 (all cells)  : {_fmt_hms(ms['gap1_all_cells'])}")
    print(f"  Oracle (ref)       : {_fmt_hms(ms['oracle'])}")
    print(f"  WALL TOTAL         : {_fmt_hms(ms['wall_total'])}  "
          f"({cs['gpu_hours']['wall_hours']:.2f} wall-hours)")
    gh = cs["gpu_hours"]
    print(f"  GPU-HOURS (billed) : {gh['gpu_hours_billed']:.2f}")
    ec = cs["estimated_cost"]
    print(f"  RATE               : {cs['rate']['gpu_hourly_rate']:.2f} "
          f"{cs['rate']['currency']}/GPU-hr")
    print(f"  ESTIMATED COST     : {ec['amount']:.2f} {ec['currency']}   "
          f"(= {ec['formula']})")
    print("=" * 90)
    print("  NOTE: dollar figure is an estimate. Edit NUM_GPUS_BILLED and")
    print("        GPU_HOURLY_RATE_USD at the top of the file for your provider.")
    print("=" * 90)


# ============================================================
#  MAIN
# ============================================================
def main():
    wall_t0 = time.time()
    phase_records = []

    if not os.path.exists(RWSE_CACHE):
        raise FileNotFoundError(f"RWSE cache missing: {RWSE_CACHE}\nRun build_rwse first.")
    blob = torch.load(RWSE_CACHE, weights_only=False)
    rwse = blob["patch_rwse"]; pe_dim = blob["rwse_steps"]
    print(f"[euclidean_fix] loaded RWSE cache | steps={pe_dim}")
    print(f"[euclidean_fix] device={DEVICE} "
          f"gpu={'?' if not torch.cuda.is_available() else torch.cuda.get_device_name(0)}")

    with Timer("build_graph", phase_records):
        data, _ = build_hetero_graph(RAW_DIR, CACHE_PATH, rebuild=False)
        data = T.ToUndirected()(data)
    active = blob["active"]
    patch_idx = build_patch_index(data, active)
    pres      = aspect_presence(data, patch_idx, active)
    label     = build_reasoning_label(data)
    ncnt = pres.sum(1); maskable = torch.nonzero(ncnt >= 2, as_tuple=False).squeeze(1)
    P = data["paper"].num_nodes
    print(f"[euclidean_fix] active={active} | MASKABLE={maskable.numel()}")

    with Timer("build_neighbors+whiten", phase_records):
        rel_neighbors = build_hetero_neighbors(data, active, max_neigh=REL_MAX_NEIGH)
        whitened_x = whiten_features(data, active)
    for k in whitened_x:
        print(f"[euclidean_fix] whitened '{k}': eff_rank {eff_rank_only(whitened_x[k]):.1f} "
              f"/ {whitened_x[k].size(1)}")

    # ---- GAP 2 first ----
    with Timer("GAP2_preanchor_rank", phase_records):
        gap2 = run_gap2_preanchor(data, active, patch_idx, pres, label, rwse, maskable, pe_dim)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ---- GAP 1 sweep ----
    print("\n" + "#" * 112)
    print("#  GAP 1: TRAINED EUCLIDEAN-COSINE JEPA — the recommended fix, end-to-end")
    print("#  sweep {raw, whitened} x {mean, relhetero};  retrieval = cosine similarity")
    print("#" * 112)
    cells = []
    for pool_mode in ["mean", REL_POOL]:
        for use_white in [False, True]:
            with Timer(f"GAP1_{pool_mode}_{'white' if use_white else 'raw'}", phase_records):
                try:
                    cells.append(run_euclidean_cell(
                        pool_mode, use_white, data, active, patch_idx, pres, label,
                        rwse, maskable, pe_dim, rel_neighbors, whitened_x,
                        full_pool=FULL_POOL))
                except Exception as e:
                    print(f"  [ERROR] cell pool={pool_mode} white={use_white} failed: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # ---- claim-only control (best config) ----
    with Timer("GAP1_claimonly_relhetero_white", phase_records):
        try:
            cells.append(run_euclidean_cell(
                REL_POOL, True, data, active, patch_idx, pres, label,
                rwse, maskable, pe_dim, rel_neighbors, whitened_x,
                full_pool=FULL_POOL, restrict_aspect="claim"))
        except Exception as e:
            print(f"  [ERROR] claim-only cell failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- oracle ceiling (training-free) ----
    oracle_sec = 0.0
    with Timer("oracle_reference", phase_records):
        t_or0 = time.time()
        try:
            oracle = oracle_retrieval_test(data, active, patch_idx, pres, maskable,
                                           whitened_x=whitened_x, use_white=True, seed=0)
            oracle_mrr = {a: oracle.get(a, {}).get("ctx_MRR", float("nan")) for a in active}
        except Exception as e:
            print(f"  [warn] oracle reference failed: {e}")
            oracle_mrr = {a: float("nan") for a in active}
        oracle_sec = time.time() - t_or0

    # ---- MASTER TABLE ----
    print("\n" + "=" * 130)
    print("  GAP 1 — EUCLIDEAN-COSINE JEPA  (mean ± std / %d seeds)" % len(SEEDS))
    print("=" * 130)
    print(f"  {'pool':<11}| {'feats':<9}| {'aspect':<8}| {'A1 acc':<11}| {'MRR':<11}| "
          f"{'H@10':<7}| {'pm_rk':<7}| {'train/seed':<11}| verdict")
    print("  " + "-" * 126)
    for c in cells:
        asp = c.get("restrict_aspect") or "all"
        tsec = c["timing"]["train_sec_mean"]
        print(f"  {c['pool_mode']:<11}| {c['features']:<9}| {asp:<8}| "
              f"{c['A1_acc'][0]:.3f}±{c['A1_acc'][1]:.2f}| "
              f"{c['MRR'][0]:.3f}±{c['MRR'][1]:.2f}| {c['Hits@10'][0]:<7.3f}| "
              f"{c['patchmean_eff_rank'][0]:<7.1f}| {tsec:>7.0f}s   | {c['verdict']}")
    print("  " + "-" * 126)
    print(f"  REFERENCE — hyperbolic pipeline (Table 2): MRR ~0.01, pm_rk ~2 (collapsed)")
    print(f"  REFERENCE — training-free oracle (Table 6): ctx_MRR = "
          f"{ {a: round(v,3) for a,v in oracle_mrr.items()} }")
    print("=" * 130)

    # ---- COST SUMMARY ----
    wall_sec = time.time() - wall_t0
    cost = build_cost_summary(phase_records, gap2, cells, oracle_sec, wall_sec)
    print_cost_summary(cost)

    # ---- save JSONs (with cost embedded) ----
    json.dump({"method": "GAP 2 — isolated pre-anchor pooled-latent effective rank "
                         "(mean-pool Euclidean-cosine JEPA, no anchors).",
               "result": {"node_eff_rank": gap2["node_eff_rank"],
                          "patchmean_eff_rank_preanchor": gap2["patchmean_eff_rank_preanchor"]},
               "timing": gap2["timing"], "cost": cost,
               "active_aspects": active, "seeds": SEEDS},
              open(PREANCHOR_RESULTS_JSON, "w"), indent=2)
    print(f"[save] {PREANCHOR_RESULTS_JSON}")

    json.dump({
        "method": "GAP 1 — trained Euclidean-cosine JEPA (no rank-2 target). "
                  "Loss = 1 - cos(pred,target) on raw latents; retrieval by cosine. "
                  "Sweep {raw,whitened} x {mean,relhetero} + claim-only control.",
        "cells": cells, "oracle_ctx_MRR": oracle_mrr,
        "full_pool": FULL_POOL, "ret_neg": RET_NEG,
        "active_aspects": active, "seeds": SEEDS,
        "maskable_papers": int(maskable.numel()), "total_papers": int(P),
        "cost": cost,
        "interpretation": "MRR >> 0.01 and pm_rk >> 2 => the hyperbolic mean-angle target "
                          "was the bottleneck; a high-rank Euclidean-cosine path recovers "
                          "retrieval toward the oracle ceiling.",
    }, open(EUC_RESULTS_JSON, "w"), indent=2)
    print(f"[save] {EUC_RESULTS_JSON}")


if __name__ == "__main__":
    main()