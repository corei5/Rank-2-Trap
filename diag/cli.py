"""Rebuttal diagnostics driver.

    python -m diag.cli --stages baseline,ridge,... --corpus papers|arxiv|both
    python -m diag.cli --probe-schema --corpus papers --out-dir /tmp/probe

Design notes
------------
* Stages in NO_CORPUS never touch the dataset, so `ledger` and `faithful` cannot
  be taken down by a corpus schema error.
* Stages always execute in CANONICAL_ORDER, not the order the user typed them,
  because `baseline` caches the banks every downstream stage consumes.
* A corpus directory with no *.json is SKIPPED with a warning, not an error.
* Cached banks (artifacts_<corpus>.pt) are rehydrated whenever they exist, so a
  resumed run does NOT retrain a baseline that is already status=ok.
* --stage-timeout bounds every stage, so a pathological stage is recorded as
  status=timeout and the run continues.
* Every run records `_meta` and `_schema`, so the provenance of each number
  (which JSON field fed which aspect, which feature source, which extractor) is
  recoverable without rerunning.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import signal
import sys
import time
import traceback
from typing import Any, Dict, List

import torch

from diag import adapters, figures
from diag.common import ResultsDB, StageResult, set_seed
from diag.stages_core import (stage_baseline, stage_gradaudit, stage_lossfloor,
                              stage_ridge)
from diag.stages_train import stage_capacity, stage_faithful, stage_posctrl
from diag.stages_task import stage_extraction, stage_hardpool, stage_poolladder
from diag.stages_sim import stage_phase
from diag.ledger import stage_ledger

# --------------------------------------------------------------------------- registry
STAGES = {
    "baseline":   stage_baseline,
    "ridge":      stage_ridge,
    "lossfloor":  stage_lossfloor,
    "gradaudit":  stage_gradaudit,
    "capacity":   stage_capacity,
    "posctrl":    stage_posctrl,
    "poolladder": stage_poolladder,
    "hardpool":   stage_hardpool,
    "phase":      stage_phase,
    "extraction": stage_extraction,
    "faithful":   stage_faithful,
    "ledger":     stage_ledger,
}

# baseline MUST precede anything reading ctx["runs"]; phase reads db["baseline"],
# db["lossfloor"] and db["extraction"], so it goes last of the corpus stages.
CANONICAL_ORDER = [
    "baseline", "ridge", "lossfloor", "gradaudit",
    "capacity", "posctrl",
    "poolladder", "hardpool", "extraction",
    "phase",
    "faithful",
    "ledger",
]

NEEDS_BASELINE = {"ridge", "lossfloor", "gradaudit", "hardpool", "poolladder", "phase"}
NO_CORPUS = {"ledger", "faithful"}
ALL_CORPUS_STAGES = [s for s in CANONICAL_ORDER if s not in NO_CORPUS]


# --------------------------------------------------------------------------- helpers
class StageTimeout(Exception):
    pass


@contextlib.contextmanager
def _time_limit(seconds: int):
    """SIGALRM-based soft timeout. No-op if seconds<=0 or platform lacks SIGALRM."""
    if not seconds or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise StageTimeout(f"stage exceeded --stage-timeout={seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def _parse_stages(raw: str) -> List[str]:
    if raw.strip() in ("all", "*"):
        want = list(CANONICAL_ORDER)
    else:
        want = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in want if s not in STAGES]
    if unknown:
        raise SystemExit(f"[cli] unknown stage(s): {unknown}\n"
                         f"[cli] known: {list(STAGES)}")
    seen, ordered = set(), []
    for s in CANONICAL_ORDER:
        if s in want and s not in seen:
            ordered.append(s); seen.add(s)
    return ordered


def _parse_ints(raw: str) -> List[int]:
    return [int(x) for x in str(raw).split(",") if str(x).strip()]


def _banner(txt: str, ch: str = "=") -> None:
    print(f"\n{ch * 70}\n  {txt}\n{ch * 70}", flush=True)


def _has_ok(db: ResultsDB, key: str) -> bool:
    """Hardened version of ResultsDB.has: tolerates non-dict payloads."""
    v = db.data.get(key)
    return isinstance(v, dict) and v.get("status") == "ok"


def _corpus_has_json(spec) -> bool:
    d = getattr(spec, "raw_dir", "")
    if not d or not os.path.isdir(d):
        return False
    if glob.glob(os.path.join(d, "*.json")):
        return True
    return bool(glob.glob(os.path.join(d, "**", "*.json"), recursive=True))


def _result_dict(out: StageResult) -> Dict[str, Any]:
    d = out.as_dict()
    d["status"] = out.status          # payload must never shadow the status
    return d


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser("diag.cli")
    ap.add_argument("--stages", default="baseline",
                    help="comma list, or 'all'. Executed in canonical order.")
    ap.add_argument("--corpus", default="papers",
                    help="papers | arxiv | comma list, e.g. papers,arxiv")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--capacity-seeds", default="0")
    ap.add_argument("--posctrl-seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-queries", type=int, default=4000,
                    help="queries for vector methods (0 = all papers)")
    ap.add_argument("--n-lex-queries", type=int, default=2000,
                    help="nested query subsample for BM25, which is ~500x "
                         "costlier per query than a dot product")
    ap.add_argument("--n-extract-docs", type=int, default=8000,
                    help="docs for the extraction stage's text probes")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--hard-pool-k", default="10,100,1000")
    ap.add_argument("--ladder", default="2,5,10,50,100,500,1000",
                    help="pool-ladder rungs; pools are built once at max(K)")
    ap.add_argument("--sim-n", type=int, default=20000,
                    help="bank size for the phase simulation")
    ap.add_argument("--paper-claimed-rho", type=float, default=0.9961,
                    help="the rho stated in the submission, simulated and "
                         "labelled CLAIMED next to the measured value")
    ap.add_argument("--limit-papers", type=int, default=0,
                    help="cap records read from disk (0 = no cap)")
    ap.add_argument("--min-chars", type=int, default=20,
                    help="an aspect field shorter than this counts as missing")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifact", default="",
                    help="explicit path to cached banks (.pt); default per-corpus")
    ap.add_argument("--stage-timeout", type=int, default=0,
                    help="seconds per stage; 0 = unlimited")
    ap.add_argument("--log-grads", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="recompute stages already marked status=ok")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--probe-schema", action="store_true",
                    help="print per-key coverage for the corpus and exit")
    ap.add_argument("--keep-going", action="store_true", default=True)
    a = ap.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    corpora = [c.strip() for c in a.corpus.split(",") if c.strip()]
    stages = _parse_stages(a.stages)
    seeds = _parse_ints(a.seeds)

    rc = 0
    all_dbs: Dict[str, Any] = {}
    summary: List[Dict[str, Any]] = []
    t_start = time.time()

    # ===================================================== per-corpus loop
    for cname in corpora:
        res_path = os.path.join(a.out_dir, f"rebuttal_{cname}.json")
        db = ResultsDB(res_path)
        all_dbs[cname] = db.data

        if a.figures_only:
            figures.render_all(db.data, os.path.join(a.out_dir, "figures"), cname)
            continue

        try:
            spec = adapters.get_spec(cname)
        except KeyError as e:
            print(f"  [error] {e}")
            rc = 1
            continue

        _banner(f"CORPUS: {cname}  ({spec.raw_dir})")

        if a.probe_schema:
            try:
                adapters.probe_schema(spec)
            except Exception as e:                                   # noqa: BLE001
                print(f"  [probe:{cname}] FAILED: {e}")
                rc = 1
            continue

        # ---- empty corpus: skip, do not fail the job -----------------
        need_corpus = any(s not in NO_CORPUS for s in stages)
        if need_corpus and not _corpus_has_json(spec):
            print(f"  [skip] corpus '{cname}': no *.json under {spec.raw_dir}.\n"
                  f"  [skip] point GJEPA_{cname.upper()}_RAW at the data, or drop "
                  f"'{cname}' from --corpus.")
            db.put("_schema", {"status": "skipped", "reason": "no *.json",
                               "raw_dir": spec.raw_dir})
            summary.append({"corpus": cname, "stage": "<data load>",
                            "status": "skipped", "err": "no *.json"})
            all_dbs[cname] = db.data
            continue

        db.put("_meta", {
            "status": "ok",
            "corpus": cname,
            "argv": " ".join(sys.argv[1:]),
            "stages": stages,
            "seeds": seeds,
            "epochs": a.epochs,
            "n_queries": a.n_queries,
            "n_lex_queries": a.n_lex_queries,
            "ladder": _parse_ints(a.ladder),
            "hard_pool_k": _parse_ints(a.hard_pool_k),
            "paper_claimed_rho": a.paper_claimed_rho,
            "stage_timeout_s": a.stage_timeout,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch": torch.__version__,
            "cuda": bool(torch.cuda.is_available()),
            "train_hook": os.environ.get("GJEPA_TRAIN_HOOK",
                                         "diag.refmodel (FALLBACK)"),
        })

        corpus, graph, X, fsrc = None, {}, None, "none"
        if need_corpus:
            try:
                corpus = adapters.load_texts(spec, limit=a.limit_papers or None,
                                             min_chars=a.min_chars)
                graph = adapters.load_graph(spec)
                feats = adapters.aspect_features(
                    spec, corpus, graph, os.path.join(a.out_dir, "_cache"))
                X, fsrc = feats["X"], feats["source"]

                if fsrc != "hetero":
                    print(f"  [WARN] feature source is '{fsrc}', NOT the trained "
                          f"model's hetero node features. Every number below "
                          f"describes this surrogate space; the ledger will say so.")

                db.put("_schema", {
                    "status": "ok",
                    "raw_dir": spec.raw_dir,
                    "n_kept": len(corpus["ids"]),
                    "n_seen": corpus["n_seen"],
                    "n_dropped": corpus["n_dropped"],
                    "n_missing_per_aspect": corpus["n_missing"],
                    "field_provenance": corpus["field_provenance"],
                    "aspects": list(spec.aspects),
                    "n_categories": len(set(corpus["categories"])),
                    "feature_source": fsrc,
                    "feature_is_surrogate": fsrc != "hetero",
                    "feature_shape": list(X.shape),
                    "extractor": spec.extractor,
                })
            except Exception as e:                                   # noqa: BLE001
                traceback.print_exc()
                db.put("_schema", {"status": "error", "err": str(e),
                                   "raw_dir": spec.raw_dir})
                summary.append({"corpus": cname, "stage": "<data load>",
                                "status": "error", "err": str(e)[:120]})
                print(f"  [error] corpus '{cname}' could not be loaded; skipping "
                      f"its {len([s for s in stages if s not in NO_CORPUS])} "
                      f"corpus stage(s)")
                rc = 1
                all_dbs[cname] = db.data
                if not any(s in NO_CORPUS for s in stages):
                    continue
                need_corpus = False
        else:
            print("  [info] selected stages need no corpus; skipping data load")

        ctx: Dict[str, Any] = {
            "spec": spec,
            "corpus": corpus,
            "graph": graph,
            "X": X,
            "feat_source": fsrc,
            "seeds": seeds,
            "capacity_seeds": _parse_ints(a.capacity_seeds),
            "posctrl_seeds": _parse_ints(a.posctrl_seeds),
            "hard_pool_k": _parse_ints(a.hard_pool_k),
            "ladder": _parse_ints(a.ladder),
            "n_queries": a.n_queries,
            "n_lex_queries": a.n_lex_queries,
            "n_extract_docs": a.n_extract_docs,
            "sim_n": a.sim_n,
            "paper_claimed_rho": a.paper_claimed_rho,
            "n_perm": a.n_perm,
            "out_dir": a.out_dir,
            "cache_dir": os.path.join(a.out_dir, "_cache"),
            "db": db.data,
            "all_dbs": all_dbs,
            "train_cfg": {"epochs": a.epochs, "loss": "l2",
                          "log_grads": a.log_grads},
        }
        os.makedirs(ctx["cache_dir"], exist_ok=True)

        # ---- rehydrate cached banks whenever they exist ---------------
        # <<< CHANGED: was `and "baseline" not in stages`, which meant a resumed
        # run with baseline already status=ok skipped the load and then retrained
        # the whole baseline just to repopulate ctx["runs"].
        art = a.artifact or os.path.join(a.out_dir, f"artifacts_{cname}.pt")
        if X is not None and os.path.isfile(art) and not (a.force and "baseline" in stages):
            try:
                ctx["runs"] = torch.load(art, map_location="cpu",
                                         weights_only=False)
                print(f"  [artifact] loaded {art} ({len(ctx['runs'])} run(s))")
            except Exception as e:                                   # noqa: BLE001
                print(f"  [warn] could not load {art}: {e}")

        # ---- stage loop ----------------------------------------------
        for s in stages:
            if s == "ledger":
                continue

            if s not in NO_CORPUS and X is None:
                print(f"  [skip] stage '{s}' needs corpus data, which failed to load")
                summary.append({"corpus": cname, "stage": s, "status": "skipped",
                                "err": "no corpus"})
                continue

            if _has_ok(db, s) and not a.force:
                print(f"  [skip] stage '{s}' already status=ok (use --force)")
                summary.append({"corpus": cname, "stage": s, "status": "cached"})
                continue

            if s in NEEDS_BASELINE and "runs" not in ctx:
                print(f"  [info] stage '{s}' needs banks; running baseline first")
                try:
                    set_seed(seeds[0])
                    with _time_limit(a.stage_timeout):
                        db.put("baseline", _result_dict(stage_baseline(ctx)))
                except Exception as e:                               # noqa: BLE001
                    traceback.print_exc()
                    db.put("baseline", {"status": "error", "err": str(e)})
                    summary.append({"corpus": cname, "stage": "baseline",
                                    "status": "error", "err": str(e)[:120]})
                    rc = 1
                    continue

            print(f"\n--- stage: {s} [{cname}] ---", flush=True)
            t0 = time.time()
            try:
                set_seed(seeds[0])
                with _time_limit(a.stage_timeout):
                    out = STAGES[s](ctx)
                db.put(s, _result_dict(out))
                summary.append({"corpus": cname, "stage": s,
                                "status": out.status,
                                "seconds": round(time.time() - t0, 1)})
            except StageTimeout as e:
                print(f"  [TIMEOUT] stage '{s}': {e}", flush=True)
                db.put(s, {"status": "timeout", "err": str(e),
                           "seconds": round(time.time() - t0, 1)})
                summary.append({"corpus": cname, "stage": s, "status": "timeout",
                                "err": str(e)[:120],
                                "seconds": round(time.time() - t0, 1)})
                rc = 1
                if not a.keep_going:
                    break
            except Exception as e:                                   # noqa: BLE001
                traceback.print_exc()
                db.put(s, {"status": "error", "err": str(e),
                           "trace": traceback.format_exc()[-2000:]})
                summary.append({"corpus": cname, "stage": s, "status": "error",
                                "err": str(e)[:120],
                                "seconds": round(time.time() - t0, 1)})
                rc = 1
                if not a.keep_going:
                    break

            if s == "baseline" and "runs" in ctx:
                try:
                    torch.save(
                        [{k: v for k, v in r.items()
                          if k in ("Q", "C", "gold", "loss_curve", "grad_log",
                                   "rank_curve", "update_ratio", "floor",
                                   "final_loss", "source")}
                         for r in ctx["runs"]], art)
                    print(f"  [artifact] saved {art}")
                except Exception as e:                               # noqa: BLE001
                    print(f"  [warn] could not save {art}: {e}")

        all_dbs[cname] = db.data
        if not a.no_figures and any(k in db.data for k in
                                    ("ridge", "lossfloor", "gradaudit", "posctrl",
                                     "poolladder", "phase", "hardpool")):
            try:
                figures.render_all(db.data,
                                   os.path.join(a.out_dir, "figures"), cname)
            except Exception as e:                                   # noqa: BLE001
                print(f"  [warn] figure rendering failed: {e}")

    # ===================================================== cross-corpus ledger
    if "ledger" in stages and not a.probe_schema and not a.figures_only:
        _banner("CLAIM LEDGER (cross-corpus)")
        for cname in corpora:
            p = os.path.join(a.out_dir, f"rebuttal_{cname}.json")
            if cname not in all_dbs and os.path.isfile(p):
                with open(p) as f:
                    try:
                        all_dbs[cname] = json.load(f)
                    except json.JSONDecodeError:
                        pass
        try:
            stage_ledger({"all_dbs": all_dbs, "out_dir": a.out_dir})
            summary.append({"corpus": "*", "stage": "ledger", "status": "ok"})
        except Exception as e:                                       # noqa: BLE001
            traceback.print_exc()
            summary.append({"corpus": "*", "stage": "ledger",
                            "status": "error", "err": str(e)[:120]})
            rc = 1

    # ===================================================== run summary
    if summary:
        _banner("RUN SUMMARY", "-")
        print(f"  {'corpus':10s} {'stage':12s} {'status':9s} {'secs':>7s}  note")
        print("  " + "-" * 66)
        for r in summary:
            print(f"  {r['corpus']:10s} {r['stage']:12s} {r['status']:9s} "
                  f"{r.get('seconds', ''):>7}  {r.get('err', '')}")
        n_err = sum(1 for r in summary if r["status"] in ("error", "timeout"))
        print(f"\n  {len(summary)} step(s), {n_err} error(s)/timeout(s), "
              f"{time.time() - t_start:.0f}s total")
        for c in corpora:
            p = os.path.join(a.out_dir, f"rebuttal_{c}.json")
            if os.path.isfile(p):
                print(f"  results: {p}")

    return rc


if __name__ == "__main__":
    sys.exit(main())