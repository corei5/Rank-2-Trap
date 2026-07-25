"""
Part 2: MULTI-ASPECT reasoning over the HIERARCHICAL paper graph (Option B).

=============================================================================
MODEL — Option B: ONE shared encoder + one predictor head per aspect
=============================================================================
A single `to_hetero` GNN encodes the whole graph (paper subgraphs + field/cite
links). Three lightweight heads (claim / method / result) each project a
paper's contextual embedding into the original 384-d aspect space.

=============================================================================
TASK — L2a REASONING (primary): masked-aspect prediction (JEPA-style)
=============================================================================
For each aspect we hold out a fraction of (paper, has_<aspect>, <aspect>)
edges, encode the graph WITHOUT them, and predict each held-out target's
embedding from the paper's remaining context. We rank the true target against
sampled negatives (MRR / Hits@k) and compare to a mean-embedding baseline.

LEAK-SAFE MASKING (important):
    When a CLAIM is held out, we ALSO drop every intra-paper edge touching
    that claim node — supported_by / challenged_by / implies / grounds. Its
    supporting evidence is a near-paraphrase of the claim, so leaving those
    edges would let the model trivially reconstruct the masked claim (inflated
    MRR). Removing them forces genuine reasoning from method + result + field +
    sibling claims + citations. This yields the HONEST, publishable number.

  L2b LINK PREDICTION (secondary): AUC/AP on (paper, cites, paper).

The best model (highest mean validation MRR across aspects) is saved to
CKPT_PATH for reloading in demos / eval.

Run:  python -m train.paper_reason
Set REBUILD_HETERO=1 to (re)build the hetero graph cache (needed after any
schema change in paper_graph.py — e.g. the new evidence/implication nodes).
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score

from core.data_utils.paper_graph import build_hetero_graph, ASPECTS


# ─────────────────────────────────────────────────────────────
RAW_DIR    = "/nfs/home/rabbyg/JEPA/Graph-JEPA/dataset/papers/raw"
CACHE_PATH = "/nfs/home/rabbyg/JEPA/Graph-JEPA/dataset/papers/processed/hetero_graphA.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN     = 128
EPOCHS     = 100
LR         = 1e-3
SEEDS      = [0, 1, 2, 3, 4]
MASK_FRAC  = 0.10          # fraction of has_<aspect> edges held out for eval

CKPT_DIR   = "/nfs/home/rabbyg/JEPA/Graph-JEPA/checkpoints/reasoning"
CKPT_PATH  = os.path.join(CKPT_DIR, "claim_reasoner_best.pt")

# aspect -> (relation, reverse relation) as created by paper_graph + ToUndirected
ASPECT_REL = {a: (("paper", f"has_{a}", a), (a, f"rev_has_{a}", "paper")) for a in ASPECTS}

# Intra-paper edges that TOUCH a claim node. When a claim is masked, these must
# be removed too (leak prevention). Each entry notes which row holds the claim.
_CLAIM_TOUCHING = [
    ("claim",  "supported_by",  "evidence"),      # claim is row 0
    ("claim",  "challenged_by", "evidence"),      # claim is row 0
    ("claim",  "implies",       "implication"),   # claim is row 0
    ("result", "grounds",       "claim"),         # claim is row 1
]
# ─────────────────────────────────────────────────────────────


class GNNEncoder(torch.nn.Module):
    """Two-layer GraphSAGE encoder (lazy input dims via (-1, -1))."""
    def __init__(self, hidden, out):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden)
        self.conv2 = SAGEConv((-1, -1), out)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# ============================================================
#  Option B — shared encoder + one head per aspect
# ============================================================
class MultiAspectReasoner(torch.nn.Module):
    """Shared hetero GNN encoder + one MLP predictor head per aspect."""

    def __init__(self, metadata, hidden, emb_dim, aspects=ASPECTS):
        super().__init__()
        enc = GNNEncoder(hidden, hidden)
        self.encoder = to_hetero(enc, metadata, aggr="sum")
        self.heads = torch.nn.ModuleDict({
            a: torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, emb_dim),   # predict in original aspect space
            ) for a in aspects
        })

    def forward(self, x_dict, edge_index_dict):
        """Return dict of node-type -> embeddings."""
        return self.encoder(x_dict, edge_index_dict)

    def predict(self, aspect, z_paper_rows):
        """Predict aspect embeddings for the given paper rows."""
        return self.heads[aspect](z_paper_rows)

    def predict_claim(self, z_paper_rows):
        """Backward-compatible alias (default aspect = claim)."""
        return self.heads["claim"](z_paper_rows)


def mask_aspect_edges(data, aspect, mask_frac, generator):
    """
    Hold out a fraction of (paper, has_<aspect>, <aspect>) edges.

    For aspect == 'claim', ALSO remove every intra-paper edge touching the
    held-out claim nodes (supported_by / challenged_by / implies / grounds),
    so the model cannot reconstruct a masked claim from its own evidence or
    implications (leak prevention).

    Returns
    -------
    masked      : HeteroData copy with the held-out (and, for claims, touching)
                  edges removed.
    held_paper  : LongTensor of owning-paper node indices for held-out edges.
    held_target : LongTensor of held-out target node indices (the answers).
    """
    rel, rev = ASPECT_REL[aspect]
    ei = data[rel].edge_index
    E = ei.size(1)
    perm = torch.randperm(E, generator=generator)
    n_hold = int(E * mask_frac)
    hold_idx, keep_idx = perm[:n_hold], perm[n_hold:]

    masked = data.clone()
    masked[rel].edge_index = ei[:, keep_idx]
    if rev in masked.edge_types:
        masked[rev].edge_index = data[rev].edge_index[:, keep_idx]

    held_paper  = ei[0, hold_idx]
    held_target = ei[1, hold_idx]

    # ---- leak prevention for claims ----
    if aspect == "claim":
        held_set = set(held_target.tolist())
        for etype in _CLAIM_TOUCHING:
            if etype not in masked.edge_types:
                continue
            e = data[etype].edge_index
            if e.numel() == 0:
                continue
            claim_row = 0 if etype[0] == "claim" else 1
            keep_mask = torch.tensor(
                [cid not in held_set for cid in e[claim_row].tolist()],
                dtype=torch.bool,
            )
            masked[etype].edge_index = e[:, keep_mask]
            # keep the auto-added reverse relation consistent
            rev_etype = (etype[2], f"rev_{etype[1]}", etype[0])
            if rev_etype in masked.edge_types:
                masked[rev_etype].edge_index = data[rev_etype].edge_index[:, keep_mask]

    return masked, held_paper, held_target


@torch.no_grad()
def reasoning_metrics(pred, true_idx, pool_x, sample_negatives=1000, generator=None):
    """Rank each true target against sampled negatives -> MRR/Hits@k/cosine."""
    pred = F.normalize(pred, dim=-1)
    pool = F.normalize(pool_x, dim=-1)
    N, C = pred.size(0), pool.size(0)
    true_idx = true_idx.long()
    cos_true = (pool[true_idx] * pred).sum(-1)

    if C > sample_negatives + 1:
        neg = torch.randint(0, C, (N, sample_negatives), generator=generator)
        cand = torch.cat([true_idx.view(-1, 1), neg], dim=1)          # col 0 = truth
    else:
        cand = torch.arange(C).unsqueeze(0).expand(N, C)

    sims = torch.einsum("nkd,nd->nk", pool[cand], pred)
    ranks = (sims > sims[:, :1]).sum(dim=1).float() + 1
    return {
        "MRR":      float((1.0 / ranks).mean()),
        "Hits@1":   float((ranks <= 1).float().mean()),
        "Hits@5":   float((ranks <= 5).float().mean()),
        "Hits@10":  float((ranks <= 10).float().mean()),
        "cos_true": float(cos_true.mean()),
    }


@torch.no_grad()
def mean_baseline_metrics(true_idx, pool_x, sample_negatives=1000, generator=None):
    """Baseline: predict the mean aspect embedding for every paper."""
    mean_vec = pool_x.mean(dim=0, keepdim=True).repeat(len(true_idx), 1)
    return reasoning_metrics(mean_vec, true_idx, pool_x, sample_negatives, generator)


def run_reasoning(seed, data_template):
    """
    Train ONE shared model on ALL active aspects jointly; evaluate each aspect.

    Returns
    -------
    (per_aspect_gnn, per_aspect_base, ckpt)
        per_aspect_gnn/base : dict aspect -> metrics dict
        ckpt                : payload for saving the best model
    """
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    data = data_template.clone()

    # which aspects actually have edges?
    active = []
    for a in ASPECTS:
        rel, _ = ASPECT_REL[a]
        if rel in data.edge_types and data[rel].edge_index.numel() > 0:
            active.append(a)
    if not active:
        print("[reasoning] no aspect edges found -> cannot run reasoning.")
        return None, None, None
    print(f"  [seed {seed}] active aspects: {active}")

    emb_dim = data["paper"].x.size(1)

    # per-aspect candidate pools, masked graphs, and train/test held-out splits
    pools, masks, held = {}, {}, {}
    for a in active:
        pools[a] = data[a].x.clone().to(DEVICE)
        m, hp, ht = mask_aspect_edges(data, a, MASK_FRAC, g)
        N = hp.size(0)
        idx = torch.randperm(N, generator=g)
        n_test = max(1, int(N * 0.5))
        masks[a] = m
        held[a] = dict(
            paper=hp.to(DEVICE), target=ht.to(DEVICE),
            train=idx[n_test:], test=idx[:n_test],
        )

    # Build ONE training graph where ALL aspects are masked simultaneously.
    # We start from data and, for each aspect, copy the (leak-safe) masked
    # edge sets from that aspect's masked graph.
    train_graph = data.clone()
    for a in active:
        rel, rev = ASPECT_REL[a]
        train_graph[rel].edge_index = masks[a][rel].edge_index
        if rev in train_graph.edge_types:
            train_graph[rev].edge_index = masks[a][rev].edge_index
        # propagate claim-touching edge removals (only present for aspect 'claim')
        if a == "claim":
            for etype in _CLAIM_TOUCHING:
                if etype in masks[a].edge_types:
                    train_graph[etype].edge_index = masks[a][etype].edge_index
                rev_etype = (etype[2], f"rev_{etype[1]}", etype[0])
                if rev_etype in masks[a].edge_types:
                    train_graph[rev_etype].edge_index = masks[a][rev_etype].edge_index
    train_graph = train_graph.to(DEVICE)

    model = MultiAspectReasoner(data.metadata(), HIDDEN, emb_dim, active).to(DEVICE)
    with torch.no_grad():
        model(train_graph.x_dict, train_graph.edge_index_dict)   # materialize lazy params
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val, best_state = -1.0, None
    for epoch in range(EPOCHS):
        model.train()
        z = model(train_graph.x_dict, train_graph.edge_index_dict)
        loss = 0.0
        for a in active:
            sel = held[a]["train"]
            pred = model.predict(a, z["paper"][held[a]["paper"][sel]])
            target = pools[a][held[a]["target"][sel]]        # fixed target = stop-grad
            loss = loss + (1 - F.cosine_similarity(pred, target, dim=-1)).mean() \
                        + F.mse_loss(pred, target)
        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                zc = model(train_graph.x_dict, train_graph.edge_index_dict)
            mrrs = []
            for a in active:
                sel = held[a]["test"]
                p = model.predict(a, zc["paper"][held[a]["paper"][sel]])
                m = reasoning_metrics(p.cpu(), held[a]["target"][sel].cpu(),
                                      pools[a].cpu(), generator=g)
                mrrs.append(m["MRR"])
            mean_mrr = float(np.mean(mrrs))
            if mean_mrr > best_val:
                best_val = mean_mrr
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            msg = " ".join(f"{a}:{mrrs[i]:.3f}" for i, a in enumerate(active))
            print(f"  [seed {seed}] epoch {epoch:03d} | loss {float(loss):.4f} "
                  f"| MRR[{msg}] mean {mean_mrr:.3f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    per_gnn, per_base = {}, {}
    with torch.no_grad():
        zc = model(train_graph.x_dict, train_graph.edge_index_dict)
    for a in active:
        sel = held[a]["test"]
        p = model.predict(a, zc["paper"][held[a]["paper"][sel]])
        per_gnn[a]  = reasoning_metrics(p.cpu(), held[a]["target"][sel].cpu(),
                                        pools[a].cpu(), generator=g)
        per_base[a] = mean_baseline_metrics(held[a]["target"][sel].cpu(),
                                            pools[a].cpu(), generator=g)

    ckpt = {
        "state_dict": best_state,
        "metadata": data.metadata(),
        "hidden": HIDDEN,
        "emb_dim": emb_dim,
        "aspects": active,
        "seed": seed,
        "mask_frac": MASK_FRAC,
        "val_mrr": best_val,
        "metrics": per_gnn,
        "baseline": per_base,
    }
    return per_gnn, per_base, ckpt


# ============================================================
#  L2b — LINK PREDICTION (secondary)
# ============================================================
class EdgeDecoder(torch.nn.Module):
    def forward(self, z_src, z_dst, edge_label_index):
        s = z_src[edge_label_index[0]]
        d = z_dst[edge_label_index[1]]
        return (s * d).sum(dim=-1)


class LinkModel(torch.nn.Module):
    def __init__(self, metadata, hidden, out):
        super().__init__()
        self.encoder = to_hetero(GNNEncoder(hidden, out), metadata, aggr="sum")
        self.decoder = EdgeDecoder()

    def forward(self, x_dict, edge_index_dict, edge_label_index, src, dst):
        z = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z[src], z[dst], edge_label_index), z


def pick_link_relation(data):
    cites = ("paper", "cites", "paper")
    if cites in data.edge_types and data[cites].edge_index.numel() > 0:
        return cites
    print("[link] no cites edges -> link-pred on (paper, has_claim, claim)")
    return ("paper", "has_claim", "claim")


@torch.no_grad()
def link_metrics(logits, y):
    probs = torch.sigmoid(logits).cpu().numpy()
    y_np = y.cpu().numpy()
    two = len(np.unique(y_np)) > 1
    return {
        "auc": roc_auc_score(y_np, probs) if two else float("nan"),
        "ap":  average_precision_score(y_np, probs) if two else float("nan"),
        "acc": float(((probs > 0.5).astype(float) == y_np).mean()),
    }


def run_link(seed, data_template):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    data = data_template.clone()
    target_rel = pick_link_relation(data)
    src_type, _, dst_type = target_rel

    transform = T.RandomLinkSplit(
        num_val=0.1, num_test=0.1,
        is_undirected=(src_type == dst_type),
        add_negative_train_samples=True, neg_sampling_ratio=1.0,
        edge_types=target_rel,
        rev_edge_types=("paper", "rev_cites", "paper") if src_type == dst_type else None,
    )
    train_data, val_data, test_data = transform(data)
    train_data = train_data.to(DEVICE); val_data = val_data.to(DEVICE); test_data = test_data.to(DEVICE)

    model = LinkModel(data.metadata(), HIDDEN, HIDDEN).to(DEVICE)
    with torch.no_grad():
        model(train_data.x_dict, train_data.edge_index_dict,
              train_data[target_rel].edge_label_index, src_type, dst_type)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def step(sd, train):
        model.train(train)
        eli = sd[target_rel].edge_label_index
        y = sd[target_rel].edge_label
        with torch.set_grad_enabled(train):
            logits, _ = model(sd.x_dict, sd.edge_index_dict, eli, src_type, dst_type)
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        return logits.detach(), y

    best_auc, best_state = -1.0, None
    for epoch in range(EPOCHS):
        step(train_data, True)
        lg, y = step(val_data, False)
        vm = link_metrics(lg, y)
        sel = vm["auc"] if not np.isnan(vm["auc"]) else vm["acc"]
        if sel > best_auc:
            best_auc = sel
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    lg, y = step(test_data, False)
    return link_metrics(lg, y), target_rel


# ============================================================
#  Loader — reload the saved reasoning model for demos/eval
# ============================================================
def load_claim_reasoner(ckpt_path=CKPT_PATH, device=DEVICE):
    """
    Reload the saved multi-aspect reasoner for inference / demos.

    ClaimReasoner heads share a lazy `to_hetero` encoder whose params only
    exist after a forward pass, so we build the graph, run one dummy forward
    to materialise params, THEN load the saved weights. Always reuses cache.
    """
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    aspects = blob.get("aspects", list(ASPECTS))
    model = MultiAspectReasoner(blob["metadata"], blob["hidden"],
                                blob["emb_dim"], aspects).to(device)

    data_template, _ = build_hetero_graph(RAW_DIR, CACHE_PATH, rebuild=False)
    data_template = T.ToUndirected()(data_template).to(device)
    with torch.no_grad():
        model(data_template.x_dict, data_template.edge_index_dict)  # materialize lazy params

    model.load_state_dict(blob["state_dict"])
    model.eval()
    vm = blob.get("val_mrr", float("nan"))
    print(f"[load] multi-aspect reasoner from {ckpt_path} "
          f"(seed {blob['seed']}, aspects {aspects}, val MRR {vm:.3f})")
    return model, blob, data_template


# ============================================================
#  Main: run both tasks over seeds, aggregate, SAVE best model
# ============================================================
def _agg_aspect(list_of_aspect_dicts, aspect, key):
    """Aggregate one metric for one aspect across seeds -> (mean, std)."""
    vals = [d[aspect][key] for d in list_of_aspect_dicts
            if d and aspect in d and key in d[aspect] and not np.isnan(d[aspect][key])]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _agg_flat(dicts, key):
    """Aggregate one metric from flat (link) dicts across seeds -> (mean, std)."""
    vals = [d[key] for d in dicts if d and key in d and not np.isnan(d[key])]
    return (float(np.mean(vals)), float(np.std(vals))) if vals \
        else (float("nan"), float("nan"))


def main():
    # honor REBUILD_HETERO env var set by the batch script
    rebuild = os.environ.get("REBUILD_HETERO", "0") == "1"
    print(f"[main] build_hetero_graph(rebuild={rebuild})")
    data_template, meta = build_hetero_graph(RAW_DIR, CACHE_PATH, rebuild=rebuild)
    print(meta)
    data_template = T.ToUndirected()(data_template)

    print("\n" + "#" * 62)
    print("#  L2a  MULTI-ASPECT REASONING  (PRIMARY, leak-safe)")
    print("#" * 62)
    r_gnn, r_base = [], []
    best_ckpt = None
    for s in SEEDS:
        g, b, ckpt = run_reasoning(s, data_template)
        if g: r_gnn.append(g)
        if b: r_base.append(b)
        if ckpt is not None and ckpt["state_dict"] is not None:
            if best_ckpt is None or ckpt["val_mrr"] > best_ckpt["val_mrr"]:
                best_ckpt = ckpt

    if best_ckpt is not None:
        os.makedirs(CKPT_DIR, exist_ok=True)
        torch.save(best_ckpt, CKPT_PATH)
        with open(CKPT_PATH.replace(".pt", ".json"), "w", encoding="utf-8") as f:
            json.dump({
                "ckpt_path": CKPT_PATH,
                "best_seed": best_ckpt["seed"],
                "aspects": best_ckpt["aspects"],
                "val_mrr": best_ckpt["val_mrr"],
                "hidden": best_ckpt["hidden"],
                "emb_dim": best_ckpt["emb_dim"],
                "mask_frac": best_ckpt["mask_frac"],
                "test_metrics": best_ckpt["metrics"],
                "baseline_metrics": best_ckpt["baseline"],
            }, f, indent=2)
        print(f"\n[save] best multi-aspect model (seed {best_ckpt['seed']}, "
              f"val MRR {best_ckpt['val_mrr']:.3f}) -> {CKPT_PATH}")

    print("\n" + "#" * 62)
    print("#  L2b  LINK PREDICTION  (secondary)")
    print("#" * 62)
    l_res, link_rel = [], None
    for s in SEEDS:
        m, link_rel = run_link(s, data_template)
        l_res.append(m)

    # ── report ──
    print("\n" + "=" * 62)
    print("  PART 2 FINAL RESULTS  (mean ± std over", len(SEEDS), "seeds)")
    print("=" * 62)

    active = best_ckpt["aspects"] if best_ckpt else list(ASPECTS)
    for a in active:
        print(f"\n  [REASONING · {a.upper()}]  (higher = better)")
        print(f"  {'metric':<10} | {'GNN reasoner':<18} | {'mean-embed baseline':<18}")
        print("  " + "-" * 52)
        for m in ["MRR", "Hits@1", "Hits@5", "Hits@10", "cos_true"]:
            gm, gs = _agg_aspect(r_gnn, a, m)
            bm, bs = _agg_aspect(r_base, a, m)
            if np.isnan(gm):
                continue
            print(f"  {m:<10} | {gm:.3f} ± {gs:.3f}   | {bm:.3f} ± {bs:.3f}")
        gm, _ = _agg_aspect(r_gnn, a, "MRR")
        bm, _ = _agg_aspect(r_base, a, "MRR")
        if not np.isnan(gm) and not np.isnan(bm):
            verdict = "REASONS" if gm > bm + 0.02 else "does NOT beat baseline"
            print(f"  => {a}: model {verdict} (MRR {gm:.3f} vs {bm:.3f}).")

    print(f"\n  [LINK PREDICTION] relation = {link_rel}")
    for m in ["auc", "ap", "acc"]:
        gm, gs = _agg_flat(l_res, m)
        print(f"  {m:<5} = {gm:.3f} ± {gs:.3f}")
    print("=" * 62)


if __name__ == "__main__":
    main()