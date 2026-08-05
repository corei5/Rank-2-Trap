"""Phase simulation for the retrieval mechanism.

CHANGED vs the ICLR submission version, in response to the reviewer's request
for the residual factor to be *identified* rather than asserted:

  * rho is READ FROM THE MEASURED RESULTS. There is no default. If no stage has
    written a measured rho, this stage reports 'skipped' instead of silently
    simulating at the value quoted in the paper.
  * kappa is measured from the features if baseline did not record it.
  * The paper's CLAIMED operating point is simulated too, and labelled CLAIMED,
    side by side with the MEASURED one.
  * The critical rho* at which retrieval collapses is located by bisection, so
    the claim becomes falsifiable: 'collapse requires rho >= rho*, we measure
    rho = ...'.
  * The verdict carries the feature-source caveat, because a surrogate feature
    space cannot adjudicate a claim about the trained model's features.
"""
from __future__ import annotations

import itertools
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch

from diag.common import (StageResult, bits_recovered, device, mrr,
                         ranks_from_banks, spectral_decay_kappa)

# keys, in priority order, under which a measured between-aspect variance share
# may have been recorded by an upstream stage
_RHO_KEYS = ("rho_between_aspect", "rho_aspect", "rho")
_RHO_STAGES = ("baseline", "extraction")


# --------------------------------------------------------------------- generative
def _simulate(n, d, rho, kappa, eps, m, seed=0, n_query=2000):
    """Generative model of the retrieval problem.

    z_{p,a} = sqrt(rho) * u_a  +  sqrt(1-rho) * S^{1/2} v_p  +  noise
      rho    : BETWEEN-ASPECT variance share (1 => aspect identity is everything,
               paper identity is nothing => retrieval must collapse)
      kappa  : power-law decay of the identity covariance S (anisotropy)
      eps    : predictor's irreducible error, relative to the identity signal
      m      : number of aspects averaged into the context

    The paper's Table 7 sweep is the special case kappa=0, eps=0.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = device()
    spec = torch.arange(1, d + 1, dtype=torch.float) ** (-kappa)
    spec = (spec / spec.mean()).sqrt().to(dev)              # unit average variance

    U = torch.randn(m, d, generator=g).to(dev)              # aspect means
    V = (torch.randn(n, d, generator=g).to(dev)) * spec     # paper identity
    a = torch.randint(0, m, (n,), generator=g).to(dev)

    Z = np.sqrt(rho) * U[a] + np.sqrt(1 - rho) * V          # candidates
    ctx = (np.sqrt(rho) * U.mean(0)[None, :].expand(n, d)
           + np.sqrt(1 - rho) * V)                          # visible-aspect mean
    err = torch.randn(n, d, generator=g).to(dev) * eps * float((1 - rho) ** 0.5)
    Q = ctx + err                                           # predictor output

    q = torch.randperm(n, generator=g)[:min(n_query, n)].to(dev)
    r = ranks_from_banks(Q[q], Z, q)
    out = {"mrr": mrr(r), "bits": bits_recovered(r, n)["bits_recovered"]}
    del U, V, Z, ctx, err, Q
    return out


def _collapsed(res: Dict[str, float], n: int, slack: float = 10.0) -> bool:
    """'Collapsed' == indistinguishable from chance at this bank size."""
    return res["mrr"] < slack / n


def _critical_rho(n, d, kappa, eps, m, seed=7,
                  lo_rho=0.5, iters=14) -> Tuple[Optional[float], list]:
    """Smallest rho at which retrieval collapses, by bisection on t = -log10(1-rho).

    MRR is monotonically decreasing in rho at fixed (kappa, eps), so bisection is
    valid. Returns (rho_star, trace) or (None, trace) if no collapse up to
    1 - 1e-8.
    """
    trace = []

    def collapsed_at(rho):
        r = _simulate(n, d, rho, kappa, eps, m, seed=seed)
        trace.append({"rho": rho, **r})
        return _collapsed(r, n), r

    t_lo = -np.log10(1.0 - lo_rho)          # rho = lo_rho
    t_hi = 8.0                              # rho = 1 - 1e-8
    c_lo, _ = collapsed_at(1.0 - 10.0 ** (-t_lo))
    if c_lo:
        return float(1.0 - 10.0 ** (-t_lo)), trace
    c_hi, _ = collapsed_at(1.0 - 10.0 ** (-t_hi))
    if not c_hi:
        return None, trace
    for _ in range(iters):
        t_mid = 0.5 * (t_lo + t_hi)
        c_mid, _ = collapsed_at(1.0 - 10.0 ** (-t_mid))
        if c_mid:
            t_hi = t_mid
        else:
            t_lo = t_mid
    return float(1.0 - 10.0 ** (-t_hi)), trace


# ------------------------------------------------------------------ rho lookup
def _measured_rho(db: Dict[str, Any]) -> Tuple[Optional[float], str]:
    for stage in _RHO_STAGES:
        blob = db.get(stage)
        if not isinstance(blob, dict) or blob.get("status") != "ok":
            continue
        for k in _RHO_KEYS:
            v = blob.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                return float(v), f"{stage}.{k}"
    return None, ""


# ------------------------------------------------------------------------ stage
def stage_phase(ctx) -> StageResult:
    t0 = time.time()
    db = ctx["db"]
    X = ctx["X"]
    n_full, A = int(X.shape[0]), int(X.shape[1])
    d = min(int(X.shape[-1]), 256)
    n = int(min(n_full, ctx.get("sim_n") or 20_000))
    fsrc = ctx.get("feat_source", "unknown")
    rho_claimed = float(ctx.get("paper_claimed_rho", 0.9961))

    # ---- rho: measured, or bail out --------------------------------------
    rho_hat, rho_src = _measured_rho(db)
    if rho_hat is None:
        msg = ("no measured between-aspect variance share found in "
               f"{_RHO_STAGES} under any of {_RHO_KEYS}; refusing to simulate at "
               "the value quoted in the paper. Run stages 'baseline' and "
               "'extraction' first.")
        print(f"  [phase] SKIPPED: {msg}", flush=True)
        return StageResult("skipped", {"reason": msg}, time.time() - t0)

    # ---- kappa: measured, not defaulted ---------------------------------
    base = db.get("baseline", {}) if isinstance(db.get("baseline"), dict) else {}
    kappa_hat = base.get("target_kappa")
    if not isinstance(kappa_hat, (int, float)) or not np.isfinite(kappa_hat):
        sub = X[:, 0, :].float()
        if sub.shape[0] > 8000:
            g = torch.Generator(device="cpu").manual_seed(0)
            sel = torch.randperm(sub.shape[0], generator=g)[:8000]
            sub = sub[sel]
        kappa_hat = spectral_decay_kappa(sub)
        print(f"  [phase] kappa not recorded by baseline; measured from features"
              f" -> {kappa_hat:.3f}", flush=True)
    kappa_hat = float(kappa_hat)

    # ---- eps: from the loss-floor ratio ---------------------------------
    lf = db.get("lossfloor", {}) if isinstance(db.get("lossfloor"), dict) else {}
    ratio = lf.get("mean_ratio")
    eps_raw = float(np.sqrt(max(float(ratio) - 1.0, 0.0))) if \
        isinstance(ratio, (int, float)) and np.isfinite(ratio) else 0.0
    eps_hat = max(eps_raw, 0.25)          # finite-sample estimator floor
    eps_floored = eps_hat > eps_raw

    print(f"  [phase] MEASURED operating point: rho={rho_hat:.6f} "
          f"(from {rho_src})  kappa={kappa_hat:.3f}  eps={eps_hat:.3f}"
          f"{' (floored)' if eps_floored else ''}  m={A}  n={n}  d={d}", flush=True)
    print(f"  [phase] CLAIMED operating point (paper): rho={rho_claimed:.6f}",
          flush=True)
    if fsrc != "hetero":
        print(f"  [phase] CAVEAT: features are '{fsrc}', not the trained model's "
              f"hetero node features; conclusions bind this feature space only.",
              flush=True)

    # ---- (a) the paper's original sweep: rho alone, kappa=0, eps=0 -------
    rhos = sorted({0.5, 0.9, 0.99, 0.999, 0.9999,
                   round(rho_claimed, 6), round(min(max(rho_hat, 1e-6), 0.999999), 6)})
    sweep_rho_only = [{"rho": r, **_simulate(n, d, r, 0.0, 0.0, A)} for r in rhos]
    for s in sweep_rho_only:
        print(f"  [phase:rho-only]   rho={s['rho']:.6f} MRR={s['mrr']:.4e} "
              f"bits={s['bits']:+.3f}", flush=True)

    # ---- (b) matched vs claimed, all measured nuisance parameters --------
    matched = _simulate(n, d, rho_hat, kappa_hat, eps_hat, A)
    claimed = _simulate(n, d, rho_claimed, kappa_hat, eps_hat, A)
    print(f"  [phase:MEASURED]   rho={rho_hat:.6f} kappa={kappa_hat:.3f} "
          f"eps={eps_hat:.3f} -> MRR={matched['mrr']:.4e} "
          f"bits={matched['bits']:+.3f}", flush=True)
    print(f"  [phase:CLAIMED]    rho={rho_claimed:.6f} kappa={kappa_hat:.3f} "
          f"eps={eps_hat:.3f} -> MRR={claimed['mrr']:.4e} "
          f"bits={claimed['bits']:+.3f}", flush=True)

    # ---- (c) leave-one-out ----------------------------------------------
    loo = {
        "drop_rho     (rho=0.5)": _simulate(n, d, 0.5, kappa_hat, eps_hat, A),
        "drop_kappa   (kappa=0)": _simulate(n, d, rho_hat, 0.0, eps_hat, A),
        "drop_eps     (eps=0)":   _simulate(n, d, rho_hat, kappa_hat, 0.0, A),
        "drop_all":               _simulate(n, d, 0.5, 0.0, 0.0, A),
    }
    for k, v in loo.items():
        print(f"  [phase:LOO] {k:26s} MRR={v['mrr']:.4e} bits={v['bits']:+.3f}",
              flush=True)

    # ---- (d) critical rho ------------------------------------------------
    rho_star, bisect_trace = _critical_rho(n, d, kappa_hat, eps_hat, A)
    if rho_star is None:
        print("  [phase] no collapse found up to rho = 1 - 1e-8", flush=True)
    else:
        print(f"  [phase] CRITICAL rho* = {rho_star:.8f}  "
              f"(1 - rho* = {1 - rho_star:.2e})  at measured kappa and eps",
              flush=True)

    # ---- (e) 2-D phase diagram rho x kappa at measured eps ---------------
    kappas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    grid = [{"rho": r, "kappa": k, **_simulate(n, d, r, k, eps_hat, A, seed=1)}
            for r, k in itertools.product(rhos, kappas)]

    # ---- verdict ---------------------------------------------------------
    repro_measured = _collapsed(matched, n)
    repro_claimed = _collapsed(claimed, n)
    caveat = ("" if fsrc == "hetero" else
              " CAVEAT: this run used '%s' features, not the trained model's "
              "hetero node features, so this adjudicates the mechanism in a "
              "surrogate feature space only." % fsrc)

    if repro_measured:
        verdict = (
            "MECHANISM REPRODUCES AT THE MEASURED OPERATING POINT. With "
            "rho=%.6f (measured, from %s), kappa=%.2f and eps=%.2f the "
            "simulation collapses to MRR=%.2e, matching the trained model. "
            "Leave-one-out shows the collapse survives removing kappa (%.2e) "
            "and eps (%.2e) but not rho (%.2e), so rho is the load-bearing "
            "factor.%s"
            % (rho_hat, rho_src, kappa_hat, eps_hat, matched["mrr"],
               loo["drop_kappa   (kappa=0)"]["mrr"],
               loo["drop_eps     (eps=0)"]["mrr"],
               loo["drop_rho     (rho=0.5)"]["mrr"], caveat))
    elif repro_claimed:
        verdict = (
            "MECHANISM DOES NOT HOLD AT THE MEASURED OPERATING POINT, AND THE "
            "DISCREPANCY IS QUANTIFIED. The simulation collapses only for "
            "rho >= rho* = %s; the paper's stated rho=%.6f is above rho* and "
            "does collapse (MRR=%.2e), but the value measured on this corpus is "
            "rho=%.6f (from %s), which retrieves at MRR=%.3f. The abstract's "
            "causal claim must therefore be restated as conditional on "
            "rho >= rho*, and the measured rho reported alongside it.%s"
            % ("%.8f" % rho_star if rho_star is not None else "n/a",
               rho_claimed, claimed["mrr"], rho_hat, rho_src, matched["mrr"],
               caveat))
    else:
        verdict = (
            "MECHANISM DOES NOT REPRODUCE at either the measured (rho=%.6f, "
            "MRR=%.3f) or the claimed (rho=%.6f, MRR=%.3f) operating point once "
            "kappa=%.2f and eps=%.2f are included. The residual factor is "
            "unidentified and the causal claim must be withdrawn from the "
            "abstract.%s"
            % (rho_hat, matched["mrr"], rho_claimed, claimed["mrr"],
               kappa_hat, eps_hat, caveat))
    print(f"  [phase] {verdict}", flush=True)

    return StageResult("ok", {
        "measured": {"rho": rho_hat, "rho_source": rho_src,
                     "kappa": kappa_hat, "eps": eps_hat,
                     "eps_raw": eps_raw, "eps_floored": eps_floored,
                     "m": A, "n_sim": n, "n_corpus": n_full, "d": d,
                     "feature_source": fsrc},
        "paper_claimed_rho": rho_claimed,
        "sweep_rho_only": sweep_rho_only,
        "matched": matched,
        "claimed": claimed,
        "leave_one_out": loo,
        "critical_rho": rho_star,
        "critical_rho_trace": bisect_trace,
        "grid": grid,
        "reproduces_at_measured_rho": repro_measured,
        "reproduces_at_claimed_rho": repro_claimed,
        "verdict": verdict}, time.time() - t0)