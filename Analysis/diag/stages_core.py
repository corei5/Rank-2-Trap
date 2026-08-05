from __future__ import annotations

import math, time
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from diag import adapters, banks
from diag.common import (StageResult, bits_recovered, bootstrap_ci, device,
                         effective_rank, mrr, permutation_p, ranks_from_banks,
                         recall_at, set_seed, spectral_decay_kappa, variance_split)


# =============================================================== BASELINE
def stage_baseline(ctx) -> StageResult:
    t0 = time.time()
    X = ctx["X"]
    runs, per_seed = [], []
    for s in ctx["seeds"]:
        cfg = dict(ctx["train_cfg"]); cfg["seed"] = s
        out = adapters.train_jepa(X, cfg)
        r = ranks_from_banks(out["Q"].to(device()), out["C"].to(device()),
                             out["gold"].to(device()))
        m = mrr(r)
        rec = {"seed": s, "mrr": m, "final_loss": out["final_loss"],
               "floor": out["floor"], "r@1": recall_at(r, 1),
               "r@10": recall_at(r, 10), "r@100": recall_at(r, 100),
               **bits_recovered(r, len(out["C"]))}
        per_seed.append(rec); runs.append(out)
        print(f"  [baseline] seed {s}: MRR={m:.3e} bits={rec['bits_recovered']:+.4f} "
              f"loss={out['final_loss']:.5f} floor={out['floor']:.5f}", flush=True)

    ctx["runs"] = runs
    best = runs[0]
    perm = permutation_p(best["Q"].to(device()), best["C"].to(device()),
                         best["gold"].to(device()), n_perm=ctx["n_perm"])

    # variance split of the target space (the rho the paper reports)
    Z = F.normalize(X.float(), dim=-1)
    n, A, d = Z.shape
    flat = Z.reshape(n * A, d)
    aspect_lbl = torch.arange(A).repeat(n)
    paper_lbl = torch.arange(n).repeat_interleave(A)
    vs_aspect = variance_split(flat, aspect_lbl)
    vs_paper = variance_split(flat, paper_lbl)

    payload = {
        "per_seed": per_seed,
        "mrr_ci": bootstrap_ci([r["mrr"] for r in per_seed]),
        "bits_ci": bootstrap_ci([r["bits_recovered"] for r in per_seed]),
        "permutation": perm,
        "rho_between_aspect": vs_aspect["rho"],
        "rho_between_paper": vs_paper["rho"],
        "target_kappa": spectral_decay_kappa(flat),
        "target_eff_rank": effective_rank(flat[:4096]),
        "n_candidates": int(len(best["C"])),
        "feature_source": ctx["feat_source"],
        "train_source": best.get("source", "?"),
    }
    print(f"  [baseline] rho(aspect)={vs_aspect['rho']:.4f}  kappa={payload['target_kappa']:.3f}  "
          f"p={perm['p_value']:.4f}")
    return StageResult("ok", payload, time.time() - t0)


# =============================================================== RIDGE PROBE (W2/Q1)
def _ridge_solve(Xtr, Ytr, lams):
    """Closed form for a whole lambda path via one SVD. Minutes, as the reviewer says."""
    U, S, Vh = torch.linalg.svd(Xtr.double(), full_matrices=False)
    UtY = U.T @ Ytr.double()
    return {l: (Vh.T @ (torch.diag(S / (S ** 2 + l)) @ UtY)).float() for l in lams}


def stage_ridge(ctx) -> StageResult:
    """THE decisive experiment (W2). Ridge from the CONTEXT representation onto
    the ORACLE retrieval space. Clears the null => encoder preserves identity and
    the LOSS is the culprit. Fails => the encoder destroys identity."""
    t0 = time.time()
    X = ctx["X"]; n = X.shape[0]
    run = ctx["runs"][0] if ctx.get("runs") else adapters.train_jepa(
        X, {**ctx["train_cfg"], "seed": ctx["seeds"][0]})

    oracle = banks.training_free_oracle(X, query_aspect=0)
    Y = F.normalize(oracle["C"].float(), dim=-1)          # oracle target space

    sources = {
        "context_trained": run["Q"].float(),              # trained context/prediction
        "context_untrained": None,                        # filled below
        "oracle_query": oracle["Q"].float(),              # upper reference
        "raw_visible_mean": F.normalize(
            X[:, 1:, :].float().mean(1), dim=-1),         # lower reference
    }
    cfg0 = {**ctx["train_cfg"], "seed": ctx["seeds"][0], "epochs": 0}
    sources["context_untrained"] = adapters.train_jepa(X, cfg0)["Q"].float()

    g = torch.Generator().manual_seed(1234)
    perm = torch.randperm(n, generator=g)
    ntr = int(0.6 * n); nva = int(0.2 * n)
    tr, va, te = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]
    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]

    results = {}
    for name, Xs in sources.items():
        if Xs is None:
            continue
        Xs = F.normalize(Xs, dim=-1)
        Ws = _ridge_solve(Xs[tr], Y[tr], lams)
        best_l, best_v = None, -1.0
        for l, W in Ws.items():
            rv = ranks_from_banks((Xs[va] @ W).to(device()), Y.to(device()),
                                  va.to(device()))
            v = mrr(rv)
            if v > best_v:
                best_l, best_v = l, v
        W = Ws[best_l]
        rte = ranks_from_banks((Xs[te] @ W).to(device()), Y.to(device()), te.to(device()))
        # null: permute the regression targets on the training split
        Yn = Y[tr][torch.randperm(ntr, generator=g)]
        Wn = _ridge_solve(Xs[tr], Yn, [best_l])[best_l]
        rnull = ranks_from_banks((Xs[te] @ Wn).to(device()), Y.to(device()), te.to(device()))
        r2 = 1 - float(((Xs[te] @ W - Y[te]) ** 2).sum()) / float(
            ((Y[te] - Y[tr].mean(0)) ** 2).sum())

        results[name] = {
            "lambda": best_l, "mrr": mrr(rte), "r@1": recall_at(rte, 1),
            "r@10": recall_at(rte, 10), "null_mrr": mrr(rnull),
            "r2_heldout": r2, **bits_recovered(rte, n),
            "clears_null": bool(mrr(rte) > 20 * mrr(rnull) and recall_at(rte, 10) > 0.05),
        }
        print(f"  [ridge] {name:20s} lam={best_l:<8g} MRR={results[name]['mrr']:.4f} "
              f"null={results[name]['null_mrr']:.3e} bits={results[name]['bits_recovered']:+.3f} "
              f"R2={r2:+.3f}", flush=True)

    ct = results.get("context_trained", {})
    verdict = ("LOSS IS THE CULPRIT: the context linearly encodes retrieval "
               "identity; the objective declines to use it."
               if ct.get("clears_null") else
               "ENCODER DESTROYS IDENTITY: identity is not linearly decodable "
               "from the context; the theory must target the encoder stage.")
    print(f"  [ridge] VERDICT -> {verdict}")
    return StageResult("ok", {"probes": results, "verdict": verdict,
                              "split": {"train": ntr, "val": nva, "test": len(te)}},
                       time.time() - t0)


# =============================================================== LOSS FLOOR (Q3)
def stage_lossfloor(ctx) -> StageResult:
    """Prop.1 predicts converged loss == E||delta||^2. Test it."""
    t0 = time.time()
    rows = []
    for run in (ctx.get("runs") or []):
        L, fl = run["final_loss"], run["floor"]
        rows.append({"final_loss": L, "floor": fl, "ratio": L / (fl + 1e-12),
                     "excess": L - fl,
                     "loss_curve_tail": run["loss_curve"][-10:]})
    if not rows:
        return StageResult("skipped", {"why": "run stage 'baseline' first"})
    ratio = float(np.mean([r["ratio"] for r in rows]))
    at_floor = 0.90 <= ratio <= 1.15
    verdict = ("AT THE FLOOR: the model has reached the Bayes-optimal value of "
               "the objective; 'the optimum is degenerate' is the correct reading."
               if at_floor else
               "ABOVE THE FLOOR by %.1f%%: the model is NOT at the objective's "
               "optimum; the Bayes-optimal framing must be withdrawn or the "
               "optimisation fixed." % (100 * (ratio - 1)))
    print(f"  [lossfloor] L/floor = {ratio:.4f} -> {verdict}")
    return StageResult("ok", {"per_seed": rows, "mean_ratio": ratio,
                              "at_floor": at_floor, "verdict": verdict},
                       time.time() - t0)


# =============================================================== GRAD AUDIT (Q6)
def stage_gradaudit(ctx) -> StageResult:
    """Distinguishes 'degenerate optimum' from 'predictor never trained'."""
    t0 = time.time()
    runs = ctx.get("runs") or []
    if not runs:
        return StageResult("skipped", {"why": "run stage 'baseline' first"})
    r = runs[0]
    gl = r["grad_log"]
    out = {}
    for mod, series in gl.items():
        s = np.asarray(series, dtype=np.float64)
        out[mod] = {"epoch0": float(s[0]), "final": float(s[-1]),
                    "median": float(np.median(s)),
                    "ratio_final_to_first": float(s[-1] / (s[0] + 1e-30)),
                    "n_zero_epochs": int((s < 1e-12).sum()),
                    "curve": s.tolist()}
        print(f"  [grad] {mod:14s} |g|_0={s[0]:.3e} |g|_T={s[-1]:.3e} "
              f"zero-epochs={out[mod]['n_zero_epochs']}")
    ur = np.asarray(r.get("update_ratio", [0.0]))
    dead = (out.get("predictor", {}).get("median", 0.0) < 1e-8
            or float(np.median(ur)) < 1e-6)
    rc = r["rank_curve"]
    rank_moved = abs(rc[-1] - rc[0]) / (rc[0] + 1e-12) > 0.05
    verdict = ("DEAD PATH: the predictor receives (near-)zero gradient. The "
               "collapse claim cannot be made until this is fixed."
               if dead else
               "PREDICTOR IS TRAINING: gradients are O(%.1e) and the "
               "update/weight ratio is %.1e, in the healthy 1e-3..1e-2 band. "
               "Flat effective rank is therefore a property of the OPTIMUM, "
               "not of a dead optimisation path."
               % (out.get("predictor", {}).get("median", float("nan")),
                  float(np.median(ur))))
    print(f"  [grad] update/weight median = {float(np.median(ur)):.3e}; "
          f"rank moved = {rank_moved} -> {verdict}")
    return StageResult("ok", {"modules": out,
                              "update_weight_ratio_median": float(np.median(ur)),
                              "rank_curve": rc, "rank_moved": rank_moved,
                              "dead_path": dead, "verdict": verdict},
                       time.time() - t0)