"""Bridge between the rebuttal harness and YOUR repo.

Three contracts. If the training hook is absent the harness falls back to
`diag.refmodel` and says so loudly, so nothing silently lies to you.

  1. CORPORA        : where each dataset lives + how to read its aspect texts
  2. load_graph     : cached hetero graph + RWSE
  3. train_jepa     : one training run -> loss curve, grad log, Protocol-R banks

Quick schema check (no GPU, ~5 s):
    python -m diag.cli --probe-schema --corpus papers --out-dir /tmp/probe
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from diag.common import device


# ============================================================ 0. PATHS / ENV
_ROOT = os.environ.get(
    "GJEPA_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

# Coarser => easier pools, finer => harder pools. `field` gives O(10-100)
# papers per class, the right granularity for the W3 hard-pool test.
_CAT_KEYS = tuple(
    k.strip() for k in
    os.environ.get("GJEPA_CAT_KEYS", "field,subfield,domain,primary_topic").split(",")
    if k.strip()
)


# ============================================================ 1. CORPUS REGISTRY
@dataclass
class CorpusSpec:
    name: str
    raw_dir: str
    proc_dir: str
    aspects: tuple = ("claim", "method", "result")
    # aspect -> ordered list of candidate JSON keys; first hit wins.
    # Nested keys via dotted path, e.g. "extracted.claims".
    field_map: Dict[str, List[str]] = field(default_factory=dict)
    id_keys: tuple = ("paper_id", "openalex_id", "source_doi", "oa_doi",
                      "id", "arxiv_id", "doi", "uid")
    cat_keys: tuple = _CAT_KEYS
    extractor: str = "unknown"     # "llm:gpt-4o-mini" | "rule" | "authors" ...
    # node-type names inside the cached hetero graph, if they differ from
    # `aspects`. Override with GJEPA_NODE_TYPES="claim,method,result".
    node_types: Optional[tuple] = None

    @property
    def hetero(self) -> str:
        return os.path.join(self.proc_dir, "hetero_graphA.pt")

    @property
    def rwse(self) -> str:
        return os.path.join(self.proc_dir, "rwse_pe.pt")

    @property
    def graph_node_types(self) -> tuple:
        env = os.environ.get("GJEPA_NODE_TYPES", "").strip()
        if env:
            return tuple(x.strip() for x in env.split(",") if x.strip())
        return self.node_types or self.aspects


# ---------------------------------------------------------------------------
# Aspect -> source fields for THIS corpus schema.
#
# Deliberately conservative. Each aspect draws only from fields that ARE that
# aspect. Fallbacks stay inside the aspect: no aspect may fall back onto
# `executive_summary`, `research_context` or any other field that mixes
# claim+method+result, because that leaks identity across aspects and inflates
# the very between-aspect variance share (rho) we are trying to measure.
# ---------------------------------------------------------------------------
_FIELDS_SCIJEPA: Dict[str, List[str]] = {
    "claim":  ["claims", "research_question_hypothesis"],
    "method": ["methodological_details", "procedures_architectures"],
    "result": ["key_results", "interpretation_implications"],
}

# Generic schema, kept as a fallback for corpora that use plain names.
_FIELDS_GENERIC: Dict[str, List[str]] = {
    "claim":  ["claims", "claim", "contributions", "key_claims"],
    "method": ["methods", "method", "approach", "methodology"],
    "result": ["results", "result", "findings", "outcomes"],
}


def _merge_fields(primary: Dict[str, List[str]],
                  extra: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out = {}
    for a, keys in primary.items():
        seen, merged = set(), []
        for k in list(keys) + list(extra.get(a, [])):
            if k not in seen:
                merged.append(k); seen.add(k)
        out[a] = merged
    return out


_FIELDS_DEFAULT = _merge_fields(_FIELDS_SCIJEPA, _FIELDS_GENERIC)


CORPORA: Dict[str, CorpusSpec] = {
    # ---- corpus A: the original, author-constructed corpus ------------------
    "papers": CorpusSpec(
        name="papers",
        raw_dir=os.environ.get("GJEPA_PAPERS_RAW", f"{_ROOT}/dataset/papers/raw"),
        proc_dir=os.environ.get("GJEPA_PAPERS_PROC",
                                f"{_ROOT}/dataset/papers/processed"),
        field_map=dict(_FIELDS_DEFAULT),
        extractor=os.environ.get("GJEPA_EXTRACTOR_A", "llm:unspecified"),
    ),
    # ---- corpus B: the arXiv corpus (W5: independent replication) -----------
    "arxiv": CorpusSpec(
        name="arxiv",
        raw_dir=os.environ.get("GJEPA_ARXIV_RAW", f"{_ROOT}/dataset/arxiv/raw"),
        proc_dir=os.environ.get("GJEPA_ARXIV_PROC",
                                f"{_ROOT}/dataset/arxiv/processed"),
        field_map=dict(_FIELDS_DEFAULT),
        extractor=os.environ.get("GJEPA_EXTRACTOR_B", "llm:unspecified"),
    ),
}


def get_spec(name: str) -> CorpusSpec:
    if name not in CORPORA:
        raise KeyError(f"unknown corpus '{name}'; known: {list(CORPORA)}")
    return CORPORA[name]


# ============================================================ 2. TEXT LOADING
def _dig(obj: Any, key: str):
    """Dotted-path lookup: 'a.b.c'."""
    cur = obj
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _maybe_json(v):
    """Fields like `keywords_json` store a JSON-encoded list inside a string."""
    if isinstance(v, str):
        s = v.strip()
        if len(s) > 1 and s[0] in "[{" and s[-1] in "]}":
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return v
    return v


def _flat(v) -> str:
    """Collapse any JSON value into whitespace-normalised text."""
    v = _maybe_json(v)
    if v is None or isinstance(v, bool):
        return ""
    if isinstance(v, str):
        return " ".join(v.split())
    if isinstance(v, (list, tuple)):
        return " ".join(x for x in (_flat(i) for i in v) if x)
    if isinstance(v, dict):
        return " ".join(x for x in (_flat(i) for i in v.values()) if x)
    return str(v).strip()


def _iter_records(raw_dir: str, limit: Optional[int] = None):
    """Yield (source_file, record). Supports *.json (object or array) and *.jsonl,
    recursively."""
    files = (sorted(glob.glob(os.path.join(raw_dir, "**", "*.json"), recursive=True))
             + sorted(glob.glob(os.path.join(raw_dir, "**", "*.jsonl"), recursive=True)))
    if not files:
        raise FileNotFoundError(
            f"no *.json / *.jsonl under {raw_dir}\n"
            f"  -> set GJEPA_PAPERS_RAW / GJEPA_ARXIV_RAW, or fix dataset layout")
    n = 0
    for fp in files:
        if fp.endswith(".jsonl"):
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield fp, rec
                    n += 1
                    if limit and n >= limit:
                        return
        else:
            with open(fp) as f:
                try:
                    obj = json.load(f)
                except json.JSONDecodeError:
                    continue
            for rec in (obj if isinstance(obj, list) else [obj]):
                yield fp, rec
                n += 1
                if limit and n >= limit:
                    return


def probe_schema(spec: CorpusSpec, n_show: int = 400) -> Dict[str, Any]:
    """Per-key coverage + mean length, so you can fix `field_map` without
    burning a GPU hour."""
    cnt: Counter = Counter()
    length: Counter = Counter()
    total = 0
    for _, r in _iter_records(spec.raw_dir, limit=n_show):
        if not isinstance(r, dict):
            continue
        total += 1
        for k, v in r.items():
            s = _flat(v)
            if s:
                cnt[k] += 1
                length[k] += len(s)

    print(f"\n  [probe:{spec.name}] {total} records sampled from {spec.raw_dir}")
    print(f"  {'key':38s} {'non-empty':>10s} {'mean chars':>11s}")
    print("  " + "-" * 62)
    for k, c in cnt.most_common():
        print(f"  {k:38s} {c / max(total,1) * 100:9.1f}% {length[k] / max(c,1):11.0f}")

    def _cov(k):
        c = cnt.get(k, 0)
        return f"{k}({c / max(total,1) * 100:.0f}%, {length.get(k,0)/max(c,1):.0f}ch)"

    print("\n  current field_map:")
    for a, keys in spec.field_map.items():
        print(f"    {a:8s} <- {', '.join(_cov(k) for k in keys)}")
    print(f"    category <- {', '.join(_cov(k) for k in spec.cat_keys)}")
    print(f"    id       <- {', '.join(_cov(k) for k in spec.id_keys)}")

    # what fraction of records would survive the all-aspects-present filter?
    ok = 0
    for _, r in _iter_records(spec.raw_dir, limit=n_show):
        if isinstance(r, dict) and all(
            any(len(_flat(_dig(r, k))) >= 20 for k in spec.field_map.get(a, [a]))
            for a in spec.aspects
        ):
            ok += 1
    print(f"\n  projected keep rate: {ok}/{total} = {ok / max(total,1) * 100:.1f}%")
    if ok / max(total, 1) < 0.5:
        print("  [!] under 50% -- widen field_map or lower --min-chars before running")
    return {"n": total, "coverage": dict(cnt), "keep_rate": ok / max(total, 1)}


def load_texts(spec: CorpusSpec, limit: Optional[int] = None,
               min_chars: int = 20) -> Dict[str, Any]:
    """Returns {ids, texts[aspect][i], categories[i], ...}. Schema-tolerant and
    loud about exactly which aspect caused a record to be dropped."""
    ids: List[str] = []
    cats: List[str] = []
    texts: Dict[str, List[str]] = {a: [] for a in spec.aspects}
    miss = {a: 0 for a in spec.aspects}
    used_key: Dict[str, Dict[str, int]] = {a: {} for a in spec.aspects}
    seen_keys: set = set()
    n_seen = n_drop = 0

    for fp, r in _iter_records(spec.raw_dir, limit=limit):
        if not isinstance(r, dict):
            continue
        n_seen += 1
        seen_keys.update(r.keys())

        pid = next((str(r[k]) for k in spec.id_keys
                    if r.get(k) not in (None, "", [], {})),
                   f"{os.path.basename(fp)}#{n_seen}")
        cat = next((_flat(r[k]).split()[0] for k in spec.cat_keys
                    if k in r and _flat(r[k])), "UNK")

        row, ok = {}, True
        for a in spec.aspects:
            s, src = "", None
            for k in spec.field_map.get(a, [a]):
                cand = _flat(_dig(r, k))
                if len(cand) >= min_chars:
                    s, src = cand, k
                    break
            if not s:
                miss[a] += 1
                ok = False
            else:
                used_key[a][src] = used_key[a].get(src, 0) + 1
            row[a] = s

        if ok:
            ids.append(pid)
            cats.append(cat)
            for a in spec.aspects:
                texts[a].append(row[a])
        else:
            n_drop += 1

    if not ids:
        raise RuntimeError(
            f"[schema] 0/{n_seen} usable records from {spec.raw_dir}\n"
            f"[schema] per-aspect misses: {miss}\n"
            f"[schema] keys seen: {sorted(seen_keys)}\n"
            f"[schema] -> run: python -m diag.cli --probe-schema "
            f"--corpus {spec.name} --out-dir /tmp/probe")

    print(f"  [texts:{spec.name}] kept {len(ids)}/{n_seen} records "
          f"(dropped {n_drop}); misses per aspect: {miss}")
    for a in spec.aspects:
        mean_len = sum(len(t) for t in texts[a]) / max(len(texts[a]), 1)
        print(f"      {a:8s} <- {used_key[a]}  (mean {mean_len:.0f} chars)")
    print(f"      categories: {len(set(cats))} distinct, "
          f"top5 {Counter(cats).most_common(5)}")

    return {"ids": ids, "texts": texts, "categories": cats,
            "n_missing": miss, "n_seen": n_seen, "n_dropped": n_drop,
            "field_provenance": used_key, "keys_seen": sorted(seen_keys)}


# ============================================================ 3. GRAPH LOADING
def load_graph(spec: CorpusSpec) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if os.path.isfile(spec.hetero):
        out["hetero"] = torch.load(spec.hetero, map_location="cpu",
                                   weights_only=False)
        print(f"  [graph] loaded {spec.hetero}")
    if os.path.isfile(spec.rwse):
        out["rwse"] = torch.load(spec.rwse, map_location="cpu",
                                 weights_only=False)
    if "hetero" not in out:
        print(f"  [warn] no hetero cache at {spec.hetero}; "
              f"using text embeddings instead (this is fine, and keeps both "
              f"corpora in the SAME feature space)")
    return out


def aspect_features(spec: CorpusSpec, corpus: Dict[str, Any],
                    graph: Dict[str, Any], cache_dir: str) -> Dict[str, Any]:
    """(n_papers, n_aspects, d) aspect feature tensor.

    Preference: (a) node features from the cached hetero graph, so we diagnose
    the SAME tensors the paper trains on; (b) sentence-transformer embeddings of
    the raw texts, cached to disk.
    """
    n, A = len(corpus["ids"]), len(spec.aspects)
    g = graph.get("hetero")

    if g is not None:
        try:
            mats = []
            for nt in spec.graph_node_types:
                x = g[nt].x
                if x is None:
                    raise ValueError(f"node type '{nt}' has no .x")
                if x.shape[0] != n:
                    raise ValueError(
                        f"node type '{nt}': {x.shape[0]} nodes but {n} kept "
                        f"records (set GJEPA_NODE_TYPES, or the graph predates "
                        f"the current field_map)")
                mats.append(x.float())
            X = torch.stack(mats, dim=1)
            print(f"  [feat] from hetero graph: {tuple(X.shape)}")
            return {"X": X, "source": "hetero"}
        except Exception as e:                                  # noqa: BLE001
            print(f"  [feat] hetero path unusable ({e}); falling back to text")

    os.makedirs(cache_dir, exist_ok=True)
    model_name = os.environ.get("GJEPA_ST_MODEL",
                                "sentence-transformers/all-mpnet-base-v2")
    tag = model_name.split("/")[-1]
    cache = os.path.join(cache_dir, f"aspect_emb_{spec.name}_{tag}_{n}.pt")
    if os.path.isfile(cache):
        X = torch.load(cache, map_location="cpu")
        if X.shape[0] == n and X.shape[1] == A:
            print(f"  [feat] text-embedding cache hit: {tuple(X.shape)}")
            return {"X": X, "source": "text-cache"}

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device=str(device()))
    mats = []
    for a in spec.aspects:
        print(f"  [feat] encoding aspect '{a}' ({n} texts) ...", flush=True)
        emb = m.encode(corpus["texts"][a], batch_size=128,
                       show_progress_bar=False, convert_to_numpy=True)
        mats.append(torch.tensor(emb, dtype=torch.float))
    X = torch.stack(mats, dim=1)
    torch.save(X, cache)
    print(f"  [feat] encoded texts -> {tuple(X.shape)} (cached at {cache})")
    return {"X": X, "source": f"text:{tag}"}


# ============================================================ 4. TRAINING HOOK
def train_jepa(X: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run ONE JEPA training run and return everything the audit needs.

    Required return keys:
      loss_curve   : list[float]            per-epoch mean training loss
      grad_log     : {module: list[float]}  per-epoch mean grad L2 norm
      rank_curve   : list[float]            effective rank of predictor output
      Q, C, gold   : Protocol-R banks after training
      floor        : float                  E||delta||^2 on this split
      final_loss   : float
      model        : nn.Module (optional)

    Default is `diag.refmodel`, a stand-in. Point at your real trainer with:
        export GJEPA_TRAIN_HOOK="train.paper_reason_gjepa:train_for_diag"
    """
    hook = os.environ.get("GJEPA_TRAIN_HOOK", "").strip()
    if hook:
        if ":" not in hook:
            raise ValueError(
                f"GJEPA_TRAIN_HOOK must be 'module.path:function', got '{hook}'")
        mod_name, fn_name = hook.split(":", 1)
        import importlib
        fn = getattr(importlib.import_module(mod_name), fn_name)
        out = fn(X, cfg)
        required = {"loss_curve", "grad_log", "Q", "C", "gold", "floor",
                    "final_loss"}
        missing = required - set(out)
        if missing:
            raise KeyError(
                f"train hook '{hook}' returned no {sorted(missing)}; "
                f"see diag/refmodel.py:train_reference_jepa for the contract")
        out.setdefault("rank_curve", [float("nan")])
        out.setdefault("update_ratio", [float("nan")])
        out.setdefault("source", hook)
        return out

    from diag.refmodel import train_reference_jepa
    out = train_reference_jepa(X, cfg)
    out["source"] = "diag.refmodel"
    return out


# ============================================================ self-test
if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "papers"
    probe_schema(get_spec(name))