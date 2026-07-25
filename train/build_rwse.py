"""
build_rwse.py  —  DYNAMIC schema analysis + TRUE RWSE positional encoding.
==========================================================================
Standalone, validate-first stage for the faithful Graph-JEPA pipeline.

WHAT IT DOES
  1. Loads the hetero paper graph.
  2. DYNAMICALLY classifies every relation in data.edge_types:
        HIERARCHY : (paper, has_*, aspect)        -> patch membership
        REASONING : edges among non-paper nodes   -> intra-paper structure
        OTHER     : paper<->paper (cites) / reverse-hierarchy
  3. Propagates paper-OWNERSHIP to all reasoning nodes (aspects+evidence+...)
     by fixed-point over the reasoning edges.
  4. Builds ONE homogeneous graph of all INTRA-PAPER reasoning edges
     (papers become disconnected components -> RWSE is local & correct).
  5. Runs PyG's AddRandomWalkPE(walk_length=RWSE_STEPS)  [canonical RWSE].
  6. scatter-maps node RWSE -> per-ASPECT PATCH RWSE  (P, RWSE_STEPS).
  7. Prints a CONNECTIVITY REPORT so you can SEE whether RWSE is informative
     or degenerate (star graph). Caches everything to RWSE_CACHE.

Run:  python -m train.build_rwse
"""

import os, json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.utils import scatter
import torch_geometric.transforms as T

from core.data_utils.paper_graph import build_hetero_graph, ASPECTS


# ─────────────────────────────────────────────────────────────
RAW_DIR    = "/nfs/home/rabbyg/JEPA/Graph-JEPA/dataset/papers/raw"
CACHE_PATH = "/nfs/home/rabbyg/JEPA/Graph-JEPA/dataset/papers/processed/hetero_graphA.pt"
RWSE_CACHE = "/nfs/home/rabbyg/JEPA/Graph-JEPA/dataset/papers/processed/rwse_pe.pt"
RWSE_STEPS = 16
# ─────────────────────────────────────────────────────────────

ASPECT_REL = {a: ("paper", f"has_{a}", a) for a in ASPECTS}


# ============================================================
#  1–2.  DYNAMIC SCHEMA ANALYSIS
# ============================================================
def analyze_schema(data, active):
    """Classify relations at runtime. No hardcoded edge names."""
    aspect_types = set(active)
    hierarchy, reasoning, other = [], [], []
    reasoning_ntypes = set(aspect_types)
    for et in data.edge_types:
        src, rel, dst = et
        if src == "paper" and dst in aspect_types and rel.startswith("has_"):
            hierarchy.append(et)
        elif src == "paper" or dst == "paper":
            other.append(et)
        else:
            reasoning.append(et)
            reasoning_ntypes.add(src); reasoning_ntypes.add(dst)
    schema = {
        "hierarchy": hierarchy, "reasoning": reasoning, "other": other,
        "reasoning_ntypes": sorted(reasoning_ntypes),
        "aspect_types": sorted(aspect_types),
    }
    print("\n" + "=" * 68)
    print("  DYNAMIC SCHEMA ANALYSIS")
    print("=" * 68)
    print("NODE TYPES:")
    for nt in data.node_types:
        print(f"    {nt:12s} : {data[nt].num_nodes:>8d} nodes")
    print("HIERARCHY edges (paper -> aspect):")
    for et in hierarchy:
        print(f"    {et} : {data[et].edge_index.size(1)}")
    print("REASONING edges (aspect/evidence intra-structure):")
    if reasoning:
        for et in reasoning:
            print(f"    {et} : {data[et].edge_index.size(1)}")
    else:
        print("    (none found)  <-- WARNING: RWSE will be degenerate (star graph)")
    print("OTHER edges (paper<->paper / reverse):")
    for et in other:
        print(f"    {et} : {data[et].edge_index.size(1)}")
    print(f"REASONING node types used for RWSE: {schema['reasoning_ntypes']}")
    print("=" * 68 + "\n")
    return schema


# ============================================================
#  3.  OWNERSHIP PROPAGATION
# ============================================================
def build_ownership(data, schema, active):
    """
    owner[nt][i] = paper id owning reasoning-node i of type nt (or -1).
    Seed aspects via hierarchy, then fixed-point over reasoning edges so evidence
    (and any other reasoning node) inherits its paper.
    """
    owner = {nt: torch.full((data[nt].num_nodes,), -1, dtype=torch.long)
             for nt in schema["reasoning_ntypes"]}
    for a in active:
        rel = ASPECT_REL[a]
        if rel in data.edge_types:
            ei = data[rel].edge_index
            owner[a][ei[1]] = ei[0]
    changed, it = True, 0
    while changed and it < 20:
        changed, it = False, it + 1
        for et in schema["reasoning"]:
            s, _, d = et
            if s not in owner or d not in owner:
                continue
            ei = data[et].edge_index
            need = (owner[d][ei[1]] < 0) & (owner[s][ei[0]] >= 0)
            if need.any():
                owner[d][ei[1][need]] = owner[s][ei[0][need]]; changed = True
            need = (owner[s][ei[0]] < 0) & (owner[d][ei[1]] >= 0)
            if need.any():
                owner[s][ei[0][need]] = owner[d][ei[1][need]]; changed = True
    # report ownership coverage
    print("OWNERSHIP coverage after propagation:")
    for nt in schema["reasoning_ntypes"]:
        o = owner[nt]; cov = float((o >= 0).float().mean()) if o.numel() else 0.0
        print(f"    {nt:12s} : {int((o>=0).sum()):>8d}/{o.numel():<8d} owned ({100*cov:.1f}%)")
    return owner


# ============================================================
#  4.  HOMOGENEOUS INTRA-PAPER REASONING GRAPH
# ============================================================
def build_homogeneous_reasoning_graph(data, schema, owner):
    """
    Concatenate all reasoning nodes into one global id space; keep only
    reasoning edges whose endpoints share the same owner paper (intra-paper).
    Returns: edge_index (2,E), gid_owner (Ntot,), base (nt->offset), Ntot.
    """
    base, running = {}, 0
    for nt in schema["reasoning_ntypes"]:
        base[nt] = running
        running += data[nt].num_nodes
    Ntot = running
    gid_owner = torch.full((Ntot,), -1, dtype=torch.long)
    for nt in schema["reasoning_ntypes"]:
        o = owner[nt]; valid = o >= 0
        idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
        gid_owner[base[nt] + idx] = o[idx]

    def gid(nt, local): return base[nt] + local
    src_all, dst_all = [], []
    for et in schema["reasoning"]:
        s, _, d = et
        if s not in owner or d not in owner:
            continue
        ei = data[et].edge_index
        so, do = owner[s][ei[0]], owner[d][ei[1]]
        keep = (so >= 0) & (do >= 0) & (so == do)      # intra-paper only
        if keep.any():
            gs, gd = gid(s, ei[0][keep]), gid(d, ei[1][keep])
            src_all += [gs, gd]; dst_all += [gd, gs]   # undirected
    if src_all:
        edge_index = torch.stack([torch.cat(src_all), torch.cat(dst_all)], 0)
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)
    return edge_index, gid_owner, base, Ntot


# ============================================================
#  5–6.  RWSE  +  MAP TO PATCH PE
# ============================================================
def compute_patch_rwse(data, schema, active, owner, edge_index, gid_owner, base, Ntot):
    """PyG AddRandomWalkPE on the homogeneous graph -> node RWSE ->
    scatter-mean over each paper's aspect-a member nodes -> patch RWSE."""
    g = Data(edge_index=edge_index, num_nodes=Ntot)
    if edge_index.size(1) == 0:
        node_pe = torch.zeros(Ntot, RWSE_STEPS)
    else:
        g = AddRandomWalkPE(walk_length=RWSE_STEPS, attr_name="rwse")(g)
        node_pe = g.rwse                                    # (Ntot, RWSE_STEPS)

    P = data["paper"].num_nodes
    pe = {}
    for a in active:
        col = torch.zeros(P, RWSE_STEPS)
        if a in owner:
            o = owner[a]; valid = o >= 0
            idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            if idx.numel() > 0:
                gids = base[a] + idx
                col = scatter(node_pe[gids], o[idx], dim=0, dim_size=P, reduce="mean")
        pe[a] = col
    return pe, node_pe


# ============================================================
#  7.  CONNECTIVITY DIAGNOSTIC (is RWSE informative or degenerate?)
# ============================================================
def connectivity_report(data, edge_index, gid_owner, pe, active):
    P = data["paper"].num_nodes
    # edges per paper subgraph
    if edge_index.size(1) > 0:
        e_paper = gid_owner[edge_index[0]]
        e_paper = e_paper[e_paper >= 0]
        deg_per_paper = torch.zeros(P)
        deg_per_paper.index_add_(0, e_paper, torch.ones(e_paper.numel()))
        deg_per_paper = deg_per_paper / 2.0                  # undirected double-count
        pct_connected = float((deg_per_paper > 0).float().mean())
        mean_edges = float(deg_per_paper.mean())
    else:
        pct_connected, mean_edges = 0.0, 0.0
    # RWSE non-triviality: how much variance does the PE actually carry?
    allpe = torch.cat([pe[a] for a in active], 0)
    pe_std = float(allpe.std())
    nonzero = float((allpe.abs().sum(1) > 1e-8).float().mean())

    print("=" * 68)
    print("  RWSE CONNECTIVITY REPORT  (READ THIS)")
    print("=" * 68)
    print(f"  papers with >=1 intra-paper reasoning edge : {100*pct_connected:.1f}%")
    print(f"  mean reasoning-edges per paper subgraph    : {mean_edges:.2f}")
    print(f"  patch-RWSE global std (signal magnitude)   : {pe_std:.4f}")
    print(f"  fraction of patches with non-zero RWSE     : {100*nonzero:.1f}%")
    if pct_connected < 0.10 or pe_std < 1e-3:
        print("  VERDICT: RWSE is NEAR-DEGENERATE on this graph (mostly stars).")
        print("           -> disclose as limitation; PE carries little structure.")
    elif pct_connected < 0.40:
        print("  VERDICT: RWSE is PARTIALLY informative (many stars, some structure).")
    else:
        print("  VERDICT: RWSE is INFORMATIVE. Good — faithful PE will help.")
    print("=" * 68 + "\n")
    return {"pct_connected": pct_connected, "mean_edges": mean_edges,
            "pe_std": pe_std, "frac_nonzero": nonzero}


# ============================================================
#  DRIVER
# ============================================================
def main():
    print("[build_rwse] loading graph ...")
    data, _ = build_hetero_graph(RAW_DIR, CACHE_PATH, rebuild=False)
    data = T.ToUndirected()(data)
    active = [a for a in ASPECTS if ASPECT_REL[a] in data.edge_types
              and data[ASPECT_REL[a]].edge_index.numel() > 0]
    print(f"[build_rwse] active aspects = {active}")

    schema = analyze_schema(data, active)
    owner  = build_ownership(data, schema, active)
    edge_index, gid_owner, base, Ntot = build_homogeneous_reasoning_graph(data, schema, owner)
    pe, node_pe = compute_patch_rwse(data, schema, active, owner,
                                     edge_index, gid_owner, base, Ntot)
    report = connectivity_report(data, edge_index, gid_owner, pe, active)

    os.makedirs(os.path.dirname(RWSE_CACHE), exist_ok=True)
    torch.save({
        "active": active,
        "rwse_steps": RWSE_STEPS,
        "patch_rwse": {a: pe[a] for a in active},    # dict aspect -> (P, RWSE_STEPS)
        "report": report,
        "schema": {k: ([list(e) for e in v] if isinstance(v, list) else v)
                   for k, v in schema.items()},
    }, RWSE_CACHE)
    print(f"[build_rwse] cached patch-RWSE -> {RWSE_CACHE}")
    print(f"[build_rwse] shape per aspect  = ({data['paper'].num_nodes}, {RWSE_STEPS})")


if __name__ == "__main__":
    main()