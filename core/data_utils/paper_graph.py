"""
Part 2: Heterogeneous paper reasoning graph (scope A) — HIERARCHICAL SUBGRAPHS.

=============================================================================
WHAT THIS BUILDS
=============================================================================
ONE big HeteroData graph in which EACH PAPER is an internal reasoning
SUBGRAPH. Instead of a flat "paper -> aspect" star, a paper's aspects are
wired to each other, mirroring scientific structure:

        method --produces--> result --grounds--> claim
        claim  --supported_by---> evidence
        claim  --challenged_by--> evidence
        claim  --implies--------> implication

All paper subgraphs share ONE graph and are connected to each other via
`in_field` (shared field hubs) and `cites` (intra-corpus citations).

Why a single big graph (not 57k separate subgraphs)?
    With PyG `to_hetero`, message passing is already LOCAL: a node only
    aggregates from its true neighbours. So typed intra-paper edges give the
    exact "subgraph" behaviour while keeping one clean HeteroData that plugs
    straight into the masked-target reasoning task. No hierarchical pooling
    needed, and the JEPA eval stays intact.

=============================================================================
NODES
=============================================================================
    paper       : MiniLM(title + executive_summary)          [1 / paper]
    claim       : MiniLM(claim.details | claim.description)   [many / paper]
    method      : MiniLM(methodological_details | procedures) [~1 / paper]
    result      : MiniLM(key_results)                         [~1 / paper]
    evidence    : MiniLM(supporting/contradicting evidence)   [many / claim]
    implication : MiniLM(claim.implications)                  [many / claim]
    field       : MiniLM(field_subfield string)               [shared hubs]

=============================================================================
EDGES
=============================================================================
  Paper-level (connect a paper to its aspects and to other papers):
    (paper,  has_claim,      claim)
    (paper,  has_method,     method)
    (paper,  has_result,     result)
    (paper,  in_field,       field)
    (paper,  cites,          paper)     # intra-corpus only; dangling dropped

  Intra-paper reasoning subgraph (aspects wired to each other):
    (claim,  supported_by,   evidence)  # from supporting_evidence
    (claim,  challenged_by,  evidence)  # from contradicting_evidence
    (claim,  implies,        implication)
    (method, produces,       result)    # imposed method->result structure
    (result, grounds,        claim)     # imposed result->claim structure

=============================================================================
RECORD FORMATS HANDLED
=============================================================================
  Format A/B : record has 'summarization' (str or dict) -> {'summary': {...}};
               claims use key 'details'.
  Format C   : summary fields live at the TOP LEVEL of the record (flat);
               claims use key 'description'.
  Skipped    : NON_SCIENTIFIC_TEXT records where summary == null, and any
               record missing a usable title/executive_summary or field label.
  PARTIAL_SCIENTIFIC_TEXT records are KEPT (they still carry a valid summary).

Run once to build+cache; the resulting .pt is reused afterwards. Set
REBUILD_HETERO=1 (see paper_reason.py) to force a rebuild when the schema
changes.
"""

import os
import gc
import glob
import json
import torch
from torch_geometric.data import HeteroData


# The reasoning aspects that become prediction-target heads (Option B).
ASPECTS = ("claim", "method", "result")

# Summary-level fields (one value per paper) -> aspect type.
SUMMARY_FIELD_TO_TYPE = {
    "methodological_details":       "method",
    "procedures_and_architectures": "method",   # format A/B
    "procedures_architectures":     "method",    # format C
    "key_results":                  "result",
}

# Claim-object sub-fields that hold the primary claim text.
CLAIM_TEXT_FIELDS = ("details", "description")   # A/B: 'details', C: 'description'


# ===========================================================================
#  Record parsing helpers (handle all formats)
# ===========================================================================
def _coerce_summary(paper):
    """
    Return (summary_dict, field_label) for any supported record format, or
    (None, None) if the record is unusable (e.g. NON_SCIENTIFIC_TEXT with
    summary == null, or missing a field label).

    Format C (flat) : the summary fields sit at the top level of `paper`.
    Format A/B      : fields are nested under paper['summarization']['summary'].
    """
    # ---- Format C (flat): summary fields at the top level ----
    if "field_subfield" in paper and "summarization" not in paper:
        label = paper.get("field_subfield")
        return (paper, label) if label else (None, None)

    # ---- Format A/B: nested under 'summarization' ----
    raw = paper.get("summarization")
    if raw is None:
        return None, None
    if isinstance(raw, str):
        try:
            outer = json.loads(raw)
        except json.JSONDecodeError:
            return None, None
    elif isinstance(raw, dict):
        outer = raw
    else:
        return None, None

    summary = outer.get("summary")
    if not summary or not isinstance(summary, dict):   # NON_SCIENTIFIC_TEXT -> null
        return None, None
    label = summary.get("field_subfield")
    return (summary, label) if label else (None, None)


def _coarse(label):
    """Collapse 'Field — Subfield (extra)' to just the coarse 'Field'."""
    for sep in (" — ", " - ", "—", " – "):
        if sep in label:
            return label.split(sep)[0].strip()
    return label.strip()


def _paper_id(paper):
    """Stable OpenAlex id for citation linking. Returns 'W123...' or None."""
    oid = paper.get("openalex_id") or paper.get("oa_doi")
    if not oid:
        return None
    return oid.rstrip("/").split("/")[-1].strip()


def _referenced_ids(paper):
    """Parse oa_referenced_works (JSON-string list of URLs) -> ['W...', ...]."""
    raw = paper.get("oa_referenced_works")
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    return [u.rstrip("/").split("/")[-1].strip()
            for u in raw if isinstance(u, str) and u.strip()]


def _pick(summary, paper, *keys):
    """Return the first non-empty string among `keys`, searching summary then
    paper. Joins list-valued fields with spaces."""
    for src in (summary, paper):
        if not isinstance(src, dict):
            continue
        for k in keys:
            v = src.get(k)
            if isinstance(v, list):
                v = " ".join(str(x) for x in v if x)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _paper_text(summary, paper):
    """Text used to embed a paper node (title + executive summary)."""
    title = _pick(summary, paper, "title", "summary_title", "paper_title")
    exec_s = _pick(summary, paper, "executive_summary")
    txt = (title + ". " + exec_s).strip(" .")
    return txt if txt else None


def _get_claims_list(summary, paper):
    """Locate the list of claim objects (may be stringified JSON)."""
    for src in (summary, paper):
        if not isinstance(src, dict):
            continue
        cl = src.get("claims")
        if isinstance(cl, str):
            try:
                cl = json.loads(cl)
            except json.JSONDecodeError:
                cl = None
        if isinstance(cl, list) and cl:
            return cl
    return []


def _claim_records(summary, paper):
    """
    Return one dict per claim:
        {text, supporting[list[str]], contradicting[list[str]], implications[list[str]]}
    Empty sub-fields become empty lists. Plain-string claims are also accepted.
    """
    out = []
    for c in _get_claims_list(summary, paper):
        if isinstance(c, str):
            if c.strip():
                out.append(dict(text=c.strip(), supporting=[],
                                contradicting=[], implications=[]))
            continue
        if not isinstance(c, dict):
            continue

        text = ""
        for k in CLAIM_TEXT_FIELDS:
            v = c.get(k)
            if isinstance(v, str) and v.strip():
                text = v.strip()
                break
        if not text:
            continue

        def as_list(key):
            v = c.get(key)
            if isinstance(v, str) and v.strip():
                return [v.strip()]
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            return []

        out.append(dict(
            text=text,
            supporting=as_list("supporting_evidence"),
            contradicting=as_list("contradicting_evidence"),
            implications=as_list("implications"),
        ))
    return out


def _aspect_text(summary, paper, aspect):
    """Single text (or None) for a summary-level aspect (method / result)."""
    for field, atype in SUMMARY_FIELD_TO_TYPE.items():
        if atype != aspect:
            continue
        t = _pick(summary, paper, field)
        if t:
            return t
    return None


def _iter_raw_records(raw_dir):
    """Yield every record dict from all *.json files (dicts or lists thereof)."""
    for fp in glob.glob(os.path.join(raw_dir, "*.json")):
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception as ex:
            print(f"[paper_graph] WARNING: skip {fp}: {ex}")
            continue
        if isinstance(data, dict):
            yield data
        elif isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    yield rec


# ===========================================================================
#  Build
# ===========================================================================
def build_hetero_graph(raw_dir, cache_path,
                       embed_model="all-MiniLM-L6-v2",
                       coarse_label=True,
                       rebuild=False):
    """
    Build (or load cached) the hierarchical multi-aspect HeteroData.

    Parameters
    ----------
    raw_dir      : directory of raw *.json paper records.
    cache_path   : path to save/load the processed .pt graph.
    embed_model  : SentenceTransformer model name for node embeddings.
    coarse_label : collapse 'Field — Subfield' to coarse 'Field' for the field node.
    rebuild      : if True, ignore any cache and rebuild from scratch.

    Returns
    -------
    (data, meta) : HeteroData and a dict of counts / id maps.
    """
    if (not rebuild) and os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        print(f"[paper_graph] loaded cache {cache_path}")
        return blob["data"], blob["meta"]

    from sentence_transformers import SentenceTransformer
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    # ---------------- PASS 1: parse (strings only, cheap) ----------------
    papers = []
    skipped = 0
    for rec in tqdm(_iter_raw_records(raw_dir), desc="Parsing"):
        summary, label = _coerce_summary(rec)
        if summary is None:
            skipped += 1
            continue
        ptext = _paper_text(summary, rec)
        if ptext is None:
            skipped += 1
            continue
        papers.append(dict(
            pid=_paper_id(rec),
            text=ptext,
            claims=_claim_records(summary, rec),
            method=_aspect_text(summary, rec, "method"),
            result=_aspect_text(summary, rec, "result"),
            field=_coarse(label) if coarse_label else label.strip(),
            refs=_referenced_ids(rec),
        ))

    if not papers:
        raise RuntimeError("No usable papers parsed. Check raw_dir / schema.")

    # ---------------- diagnostic coverage report ----------------
    n_claim_p    = sum(1 for p in papers if p["claims"])
    n_method_p   = sum(1 for p in papers if p["method"])
    n_result_p   = sum(1 for p in papers if p["result"])
    total_claims = sum(len(p["claims"]) for p in papers)
    total_evid   = sum(len(c["supporting"]) + len(c["contradicting"])
                       for p in papers for c in p["claims"])
    total_impl   = sum(len(c["implications"]) for p in papers for c in p["claims"])
    print(f"[paper_graph] papers={len(papers)} skipped={skipped}")
    print(f"[paper_graph] coverage -> "
          f"claims:{n_claim_p}/{len(papers)} ({total_claims}) | "
          f"method:{n_method_p} | result:{n_result_p} | "
          f"evidence:{total_evid} | implications:{total_impl}", flush=True)
    for name, n in [("claim", n_claim_p), ("method", n_method_p),
                    ("result", n_result_p)]:
        if n == 0:
            print(f"[paper_graph] ★ WARNING: 0 '{name}' found — check field names/format.")
    if total_evid == 0:
        print("[paper_graph] ★ WARNING: 0 evidence nodes — supporting/contradicting"
              " fields missing; subgraph will lack evidence edges.")

    # ---------------- id maps ----------------
    paper_row = {}
    for i, p in enumerate(papers):
        if p["pid"] and p["pid"] not in paper_row:
            paper_row[p["pid"]] = i
    fields_sorted = sorted({p["field"] for p in papers})
    field_map = {f: i for i, f in enumerate(fields_sorted)}

    # ---------------- flatten nodes + intra-paper edges ----------------
    # claims (many per paper)
    claim_texts, claim_owner = [], []
    # evidence
    evid_texts = []
    ev_support_src, ev_support_dst = [], []      # (claim, supported_by, evidence)
    ev_contra_src,  ev_contra_dst  = [], []      # (claim, challenged_by, evidence)
    # implications
    impl_texts = []
    impl_src, impl_dst = [], []                  # (claim, implies, implication)

    for i, p in enumerate(papers):
        for c in p["claims"]:
            cidx = len(claim_texts)
            claim_texts.append(c["text"])
            claim_owner.append(i)
            for e in c["supporting"]:
                ev_support_src.append(cidx)
                ev_support_dst.append(len(evid_texts))
                evid_texts.append(e)
            for e in c["contradicting"]:
                ev_contra_src.append(cidx)
                ev_contra_dst.append(len(evid_texts))
                evid_texts.append(e)
            for im in c["implications"]:
                impl_src.append(cidx)
                impl_dst.append(len(impl_texts))
                impl_texts.append(im)

    # method / result (one per paper that has it)
    method_texts, method_owner = [], []
    result_texts, result_owner = [], []
    for i, p in enumerate(papers):
        if p["method"]:
            method_owner.append(i)
            method_texts.append(p["method"])
        if p["result"]:
            result_owner.append(i)
            result_texts.append(p["result"])

    # intra-paper structural edges: method->produces->result, result->grounds->claim
    method_row = {mo: k for k, mo in enumerate(method_owner)}   # paper idx -> method node idx
    result_row = {ro: k for k, ro in enumerate(result_owner)}   # paper idx -> result node idx
    prod_src, prod_dst = [], []                                 # (method, produces, result)
    for pi in range(len(papers)):
        if pi in method_row and pi in result_row:
            prod_src.append(method_row[pi])
            prod_dst.append(result_row[pi])
    grnd_src, grnd_dst = [], []                                 # (result, grounds, claim)
    for cidx, owner in enumerate(claim_owner):
        if owner in result_row:
            grnd_src.append(result_row[owner])
            grnd_dst.append(cidx)

    # ---------------- PASS 2: embed ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(embed_model, device=device)

    def embed(texts, prog=False):
        if not texts:
            return torch.empty((0, emb_dim), dtype=torch.float)
        arr = model.encode(texts, batch_size=256, convert_to_numpy=True,
                            show_progress_bar=prog)
        return torch.tensor(arr, dtype=torch.float)

    # paper embeddings first (defines emb_dim used by empty tensors above)
    paper_x = torch.tensor(
        model.encode([p["text"] for p in papers], batch_size=256,
                     convert_to_numpy=True, show_progress_bar=True),
        dtype=torch.float,
    )
    emb_dim = paper_x.size(1)

    field_x  = embed(fields_sorted)
    claim_x  = embed(claim_texts, prog=True)
    method_x = embed(method_texts, prog=True)
    result_x = embed(result_texts, prog=True)
    evid_x   = embed(evid_texts, prog=True)
    impl_x   = embed(impl_texts, prog=True)

    del model
    gc.collect()

    # ---------------- edge builders ----------------
    def owner_edge(owner_list):
        """(owner_paper_idx -> sequential_node_idx) edge tensor."""
        src = torch.tensor(owner_list, dtype=torch.long) if owner_list \
              else torch.empty(0, dtype=torch.long)
        dst = torch.arange(len(owner_list), dtype=torch.long)
        return torch.stack([src, dst], 0)

    def pair_edge(src, dst):
        s = torch.tensor(src, dtype=torch.long) if src else torch.empty(0, dtype=torch.long)
        d = torch.tensor(dst, dtype=torch.long) if dst else torch.empty(0, dtype=torch.long)
        return torch.stack([s, d], 0)

    inf_src = torch.arange(len(papers), dtype=torch.long)
    inf_dst = torch.tensor([field_map[p["field"]] for p in papers], dtype=torch.long)

    cites_src, cites_dst, n_dangling = [], [], 0
    for i, p in enumerate(papers):
        for r in p["refs"]:
            j = paper_row.get(r)
            if j is not None:
                cites_src.append(i)
                cites_dst.append(j)
            else:
                n_dangling += 1
    print(f"[paper_graph] intra-corpus cites edges={len(cites_src)} "
          f"dangling(dropped)={n_dangling}", flush=True)
    if not cites_src:
        print("[paper_graph] ★ WARNING: NO intra-corpus citation edges. "
              "Reasoning relies on subgraph + field structure only.")

    # ---------------- assemble HeteroData ----------------
    data = HeteroData()
    data["paper"].x       = paper_x
    data["field"].x       = field_x
    data["claim"].x       = claim_x
    data["method"].x      = method_x
    data["result"].x      = result_x
    data["evidence"].x    = evid_x
    data["implication"].x = impl_x
    data["paper"].field_y = inf_dst.clone()   # convenience field label per paper

    # paper-level edges
    data["paper", "has_claim",  "claim"].edge_index  = owner_edge(claim_owner)
    data["paper", "has_method", "method"].edge_index = owner_edge(method_owner)
    data["paper", "has_result", "result"].edge_index = owner_edge(result_owner)
    data["paper", "in_field",   "field"].edge_index  = torch.stack([inf_src, inf_dst], 0)
    data["paper", "cites",      "paper"].edge_index  = pair_edge(cites_src, cites_dst)

    # intra-paper reasoning subgraph edges
    data["claim",  "supported_by",  "evidence"].edge_index    = pair_edge(ev_support_src, ev_support_dst)
    data["claim",  "challenged_by", "evidence"].edge_index    = pair_edge(ev_contra_src,  ev_contra_dst)
    data["claim",  "implies",       "implication"].edge_index = pair_edge(impl_src, impl_dst)
    data["method", "produces",      "result"].edge_index      = pair_edge(prod_src, prod_dst)
    data["result", "grounds",       "claim"].edge_index       = pair_edge(grnd_src, grnd_dst)

    meta = dict(
        field_map=field_map,
        aspects=list(ASPECTS),
        num_papers=len(papers),
        num_claims=len(claim_texts),
        num_methods=len(method_texts),
        num_results=len(result_texts),
        num_evidence=len(evid_texts),
        num_implications=len(impl_texts),
        num_fields=len(fields_sorted),
        emb_dim=emb_dim,
        skipped=skipped,
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"data": data, "meta": meta}, cache_path)
    print(f"[paper_graph] saved cache -> {cache_path}")
    print(f"[paper_graph] nodes: paper={meta['num_papers']} claim={meta['num_claims']} "
          f"method={meta['num_methods']} result={meta['num_results']} "
          f"evidence={meta['num_evidence']} impl={meta['num_implications']} "
          f"field={meta['num_fields']} | emb_dim={meta['emb_dim']}")
    return data, meta