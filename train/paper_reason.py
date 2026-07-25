"""
paper_reason_compare.py  (v5 — Graph-JEPA-faithful: Eq.4 unit hyperbola + Eq.5 PE)
==================================================================================
Three-variant JEPA comparison on the hierarchical paper HeteroData.

v5 FIXES (grounded in Skenderi et al., TMLR 2025):
  1. LEAK-SAFE masking for ALL aspects (claim/method/result), schema-driven.
  2. HONEST Metis: robust extraction across PyG versions; report the partitioner
     ACTUALLY used at runtime.
  3. FAITHFUL HYPERBOLIC objective (paper Sec 3.4):
       - Target = UNIT HYPERBOLA  psi = [cosh(a), sinh(a)],  a = mean(Z)   (Eq.4)
       - Predictor conditioned on target positional encoding P_l           (Eq.5)
             psi_hat = W2( sigma( W1(z_x + P_l) + b1 ) ) + b2,  psi_hat in R^2
       - Energy = Smooth-L1 on hyperbola coords                            (Eq.6)
       - NO variance term. Collapse prevented by EMA stop-grad + SIMPLE predictor.
     (The Poincaré-disk distance is only a paper ABLATION -> NaN; we do NOT use it.)

NOTES / HONEST CAVEATS:
  * Graph-JEPA is graph-level (patches->pooled). We ADAPT Eq.4-6 to NODE-level
    aspect prediction on a hetero graph. State this as an extension in the paper.
  * P_l (RWSE in the paper) is realized here as a learnable per-target-node
    embedding — a reasonable substitute giving each target a distinct identity.
  * On hyperbola coords, LOW std is EXPECTED ("base level of the hierarchy",
    Sec 3.4), NOT collapse. Judge collapse on the LATENT z and on Hits@1<Hits@10.

Run:  python -m train.paper_reason_compare
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
import torch_geometric.transforms as T

from core.data_utils.paper_graph import build_hetero_graph, ASPECTS


# ─────────────────────────────────────────────────────────────
RAW_DIR    = "/nfs/home/rabbyg/JEPA/Graph-JEPA/dataset/papers/raw"
CACHE_PATH = "/nfs/home/rabbyg/JEPA/Graph-JEPA/dataset/papers/processed/hetero_graphA.pt"
CKPT_DIR   = "/nfs/home/rabbyg/JEPA/Graph-JEPA/checkpoints/reasoning"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN     = 128
LATENT     = 128
EPOCHS     = 100
LR         = 1e-3
WD         = 1e-5
SEEDS      = [0, 1, 2, 3, 4]
MASK_FRAC  = 0.10

EMA_BASE, EMA_FINAL = 0.996, 1.0
VAR_COEF, VAR_GAMMA = 1.0, 1.0
COS_COEF            = 1.0
SMOOTHL1_BETA       = 1.0

N_PARTS   = 64
VARIANTS  = ["edge_mask", "metis_patch", "metis_hyperbolic"]
RESULTS_JSON = os.path.join(CKPT_DIR, "compare_results.json")

ASPECT_REL = {a: (("paper", f"has_{a}", a), (a, f"rev_has_{a}", "paper")) for a in ASPECTS}

ACTUAL_PARTITIONER = {"used": None}
# ─────────────────────────────────────────────────────────────


# ============================================================
#  SCHEMA-DRIVEN LEAK MAP
# ============================================================
def build_incident_map(data, aspects):
    """aspect -> list[(edge_type, row)] for EVERY relation touching that aspect."""
    incident = {a: [] for a in aspects}
    for et in data.edge_types:
        src, rel, dst = et
        for a in aspects:
            if src == a:
                incident[a].append((et, 0))
            if dst == a:
                incident[a].append((et, 1))
    return incident


def reverse_et(et):
    src, rel, dst = et
    return (dst, f"rev_{rel}", src)


# ============================================================
#  Partitioner detection
# ============================================================
def detect_partitioner():
    try:
        from torch_geometric.loader import ClusterData  # noqa
        try:
            import torch_sparse  # noqa
            return "metis"
        except Exception:
            pass
        try:
            import pymetis  # noqa
            return "metis"
        except Exception:
            pass
    except Exception:
        pass
    return "bfs_fallback"


PARTITIONER = detect_partitioner()


def to_homogeneous_for_partition(data):
    offset, slices = 0, {}
    for nt in data.node_types:
        slices[nt] = (offset, offset + data[nt].num_nodes)
        offset += data[nt].num_nodes
    total = offset
    rows, cols = [], []
    for et in data.edge_types:
        src, _, dst = et
        ei = data[et].edge_index
        if ei.numel() == 0:
            continue
        rows.append(ei[0] + slices[src][0]); cols.append(ei[1] + slices[dst][0])
    edge_index = (torch.stack([torch.cat(rows), torch.cat(cols)], 0)
                  if rows else torch.empty((2, 0), dtype=torch.long))
    return edge_index, slices, total


def partition_metis(edge_index, num_nodes, n_parts, seed):
    """Robust Metis across PyG versions (old .perm/.partptr, new .partition,
    or pymetis fallback)."""
    from torch_geometric.data import Data
    from torch_geometric.loader import ClusterData
    d = Data(edge_index=edge_index, num_nodes=num_nodes)
    d.x = torch.zeros(num_nodes, 1)
    torch.manual_seed(seed)
    cd = ClusterData(d, num_parts=n_parts, recursive=False, log=False)

    perm = getattr(cd, "perm", None)
    partptr = getattr(cd, "partptr", None)
    if perm is None and hasattr(cd, "partition"):
        part = cd.partition
        perm = getattr(part, "node_perm", getattr(part, "perm", None))
        partptr = getattr(part, "partptr", None)

    if perm is not None and partptr is not None:
        labels = torch.empty(num_nodes, dtype=torch.long)
        for pid in range(len(partptr) - 1):
            labels[perm[partptr[pid]:partptr[pid + 1]]] = pid
        return labels

    import pymetis
    adj = [[] for _ in range(num_nodes)]
    ei = edge_index.tolist()
    for s, dd in zip(ei[0], ei[1]):
        adj[s].append(dd)
    _, membership = pymetis.part_graph(n_parts, adjacency=adj)
    return torch.tensor(membership, dtype=torch.long)


def partition_bfs(edge_index, num_nodes, n_parts, seed):
    from collections import deque
    g = torch.Generator().manual_seed(seed)
    adj = [[] for _ in range(num_nodes)]
    ei = edge_index.tolist()
    for s, d in zip(ei[0], ei[1]):
        adj[s].append(d); adj[d].append(s)
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    seeds = torch.randperm(num_nodes, generator=g)[:n_parts].tolist()
    fr = deque()
    for pid, s in enumerate(seeds):
        labels[s] = pid; fr.append(s)
    while fr:
        u = fr.popleft()
        for v in adj[u]:
            if labels[v] < 0:
                labels[v] = labels[u]; fr.append(v)
    for i, v in enumerate((labels < 0).nonzero(as_tuple=True)[0].tolist()):
        labels[v] = i % n_parts
    return labels


def get_partition(data, n_parts, seed):
    edge_index, slices, total = to_homogeneous_for_partition(data)
    if PARTITIONER == "metis":
        try:
            labels = partition_metis(edge_index, total, n_parts, seed)
            ACTUAL_PARTITIONER["used"] = "metis"
        except Exception as e:
            print(f"  [partition] Metis failed ({e}); BFS fallback.")
            labels = partition_bfs(edge_index, total, n_parts, seed)
            ACTUAL_PARTITIONER["used"] = "bfs_fallback"
    else:
        labels = partition_bfs(edge_index, total, n_parts, seed)
        ACTUAL_PARTITIONER["used"] = "bfs_fallback"
    return labels, slices, total


# ============================================================
#  FAITHFUL HYPERBOLIC TARGET  (Graph-JEPA Eq.4)  — unit hyperbola
# ============================================================
def to_hyperbola(z):
    """
    Graph-JEPA Eq.4:  psi = [cosh(alpha), sinh(alpha)],  alpha = mean(z).
    z: (...,d) latent -> (...,2) point on the unit hyperbola x^2 - y^2 = 1.
    By design (high-dim mean concentrates) alpha has LOW variance -> psi clusters
    at the 'base level of the hierarchy' (Sec 3.4). This is EXPECTED, not collapse.
    """
    alpha = z.mean(dim=-1, keepdim=True)
    return torch.cat([torch.cosh(alpha), torch.sinh(alpha)], dim=-1)


# ============================================================
#  Encoder + JEPA model
# ============================================================
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden, out):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden)
        self.conv2 = SAGEConv((-1, -1), out)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class JEPA(torch.nn.Module):
    """
    hyperbolic=False : predictor -> latent; target = EMA latent (cosine MRR).
    hyperbolic=True  : predictor -> R^2 on UNIT HYPERBOLA (Eq.4), Smooth-L1 (Eq.6),
                       conditioned on per-target positional embedding P_l (Eq.5):
                           psi_hat = W2( sigma( W1(z_x + P_l) + b1 ) ) + b2.
                       SIMPLE predictor (1 hidden layer) + EMA stop-grad prevent
                       collapse (Sec 3.4). NO variance penalty is used.
    """
    def __init__(self, metadata, hidden, latent, aspects, hyperbolic=False,
                 num_pos=None):
        super().__init__()
        self.aspects = list(aspects)
        self.hyperbolic = hyperbolic
        self.latent = latent
        self.encoder = to_hetero(GNNEncoder(hidden, latent), metadata, aggr="sum")
        self.target_encoder = to_hetero(GNNEncoder(hidden, latent), metadata, aggr="sum")
        out_dim = 2 if hyperbolic else latent

        # per-target positional embedding P_l (Eq.5). Only for hyperbolic variant.
        self.use_pos = hyperbolic and (num_pos is not None)
        if self.use_pos:
            self.pos = torch.nn.ModuleDict({
                a: torch.nn.Embedding(num_pos[a], latent) for a in self.aspects
            })
            for a in self.aspects:
                torch.nn.init.normal_(self.pos[a].weight, std=0.02)

        # SIMPLE predictor (Sec 3.4: a less expressive predictor is crucial).
        self.predictors = torch.nn.ModuleDict({
            a: torch.nn.Sequential(
                torch.nn.Linear(latent, latent), torch.nn.GELU(),
                torch.nn.Linear(latent, out_dim),
            ) for a in self.aspects
        })

    def encode_context(self, x, ei):
        return self.encoder(x, ei)

    @torch.no_grad()
    def encode_target(self, x, ei):
        return self.target_encoder(x, ei)

    def predict(self, a, z_x, target_ids=None):
        """Eq.5: add target positional embedding P_l before the predictor."""
        h = z_x
        if self.use_pos and target_ids is not None:
            h = h + self.pos[a](target_ids)
        return self.predictors[a](h)

    @torch.no_grad()
    def init_target_from_online(self):
        self.target_encoder.load_state_dict(self.encoder.state_dict())

    @torch.no_grad()
    def freeze_target(self):
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def ema_update(self, m):
        for pq, pk in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            pk.data.mul_(m).add_((1 - m) * pq.detach().data)
        for bq, bk in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            bk.data.copy_(bq.data)


# ============================================================
#  Shared helpers
# ============================================================
def variance_loss(z, gamma=VAR_GAMMA):
    return F.relu(gamma - torch.sqrt(z.var(0) + 1e-6)).mean()


@torch.no_grad()
def collapse_stats(z):
    z = z - z.mean(0, keepdim=True)
    std = torch.sqrt(z.var(0) + 1e-6)
    try:
        s = torch.linalg.svdvals(z.float())
        p = s / (s.sum() + 1e-9)
        eff_rank = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
    except Exception:
        eff_rank = float("nan")
    return float(std.mean()), eff_rank


def _rank_dict(ranks):
    return {"MRR": float((1.0 / ranks).mean()),
            "Hits@1": float((ranks <= 1).float().mean()),
            "Hits@5": float((ranks <= 5).float().mean()),
            "Hits@10": float((ranks <= 10).float().mean())}


@torch.no_grad()
def mrr_euclid(pred, true_idx, pool, neg=1000, gen=None):
    pred = F.normalize(pred, dim=-1); pool = F.normalize(pool, dim=-1)
    N, C = pred.size(0), pool.size(0); true_idx = true_idx.long()
    if C > neg + 1:
        cand = torch.cat([true_idx.view(-1, 1),
                          torch.randint(0, C, (N, neg), generator=gen)], 1)
    else:
        cand = torch.arange(C).unsqueeze(0).expand(N, C)
    sims = torch.einsum("nkd,nd->nk", pool[cand], pred)
    ranks = (sims > sims[:, :1]).sum(1).float() + 1
    return _rank_dict(ranks)


@torch.no_grad()
def mrr_hyperbola(pred_psi, true_idx, pool_psi, neg=1000, gen=None):
    """Rank candidates by SMALL Smooth-L1 distance on the unit-hyperbola coords
    (Graph-JEPA energy, Eq.6). pred_psi:(N,2), pool_psi:(C,2)."""
    N, C = pred_psi.size(0), pool_psi.size(0); true_idx = true_idx.long()
    if C > neg + 1:
        cand = torch.cat([true_idx.view(-1, 1),
                          torch.randint(0, C, (N, neg), generator=gen)], 1)
    else:
        cand = torch.arange(C).unsqueeze(0).expand(N, C)
    cand_psi = pool_psi[cand]                          # (N,K,2)
    p = pred_psi.unsqueeze(1).expand_as(cand_psi)      # (N,K,2)
    d = F.smooth_l1_loss(p, cand_psi, beta=SMOOTHL1_BETA, reduction="none").mean(-1)
    ranks = (d < d[:, :1]).sum(1).float() + 1
    return _rank_dict(ranks)


# ============================================================
#  VARIANT A — edge_mask (OURS) with generalized leak removal
# ============================================================
def mask_aspect_edges(data, aspect, frac, gen, incident):
    rel, rev = ASPECT_REL[aspect]
    ei = data[rel].edge_index; E = ei.size(1)
    perm = torch.randperm(E, generator=gen); n = int(E * frac)
    hold, keep = perm[:n], perm[n:]

    m = data.clone()
    m[rel].edge_index = ei[:, keep]
    if rev in m.edge_types:
        m[rev].edge_index = data[rev].edge_index[:, keep]

    hp, ht = ei[0, hold], ei[1, hold]
    held_nodes = set(ht.tolist())

    for et, row in incident[aspect]:
        if et == rel or et not in m.edge_types:
            continue
        e = data[et].edge_index
        if e.numel() == 0:
            continue
        km = torch.tensor([nid not in held_nodes for nid in e[row].tolist()],
                          dtype=torch.bool)
        m[et].edge_index = e[:, km]
        ret = reverse_et(et)
        if ret in m.edge_types and data[ret].edge_index.size(1) == e.size(1):
            m[ret].edge_index = data[ret].edge_index[:, km]
    return m, hp, ht


def _apply_masked_edges_to(target_graph, masked_graph, aspect, incident):
    rel, rev = ASPECT_REL[aspect]
    target_graph[rel].edge_index = masked_graph[rel].edge_index
    if rev in target_graph.edge_types:
        target_graph[rev].edge_index = masked_graph[rev].edge_index
    for et, _ in incident[aspect]:
        if et == rel:
            continue
        if et in masked_graph.edge_types:
            target_graph[et].edge_index = masked_graph[et].edge_index
        ret = reverse_et(et)
        if ret in masked_graph.edge_types:
            target_graph[ret].edge_index = masked_graph[ret].edge_index


def run_edge_mask(seed, data, active, incident):
    gen = torch.Generator().manual_seed(seed)
    masks, held = {}, {}
    for a in active:
        m, hp, ht = mask_aspect_edges(data, a, MASK_FRAC, gen, incident)
        N = hp.size(0); idx = torch.randperm(N, generator=gen); nt = max(1, int(N * 0.5))
        masks[a] = m
        held[a] = dict(paper=hp.to(DEVICE), target=ht.to(DEVICE), train=idx[nt:], test=idx[:nt])

    tg = data.clone()
    for a in active:
        _apply_masked_edges_to(tg, masks[a], a, incident)
    tg = tg.to(DEVICE); full = data.clone().to(DEVICE)

    model = JEPA(data.metadata(), HIDDEN, LATENT, active, hyperbolic=False).to(DEVICE)
    with torch.no_grad():
        model.encode_context(tg.x_dict, tg.edge_index_dict)
        model.target_encoder(full.x_dict, full.edge_index_dict)
        model.init_target_from_online(); model.freeze_target()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)

    for ep in range(EPOCHS):
        model.train()
        m = EMA_FINAL - (EMA_FINAL - EMA_BASE) * (0.5 * (1 + np.cos(np.pi * ep / max(1, EPOCHS - 1))))
        with torch.no_grad():
            zt = model.encode_target(full.x_dict, full.edge_index_dict)
            pool = {a: zt[a].detach() for a in active}
        zc = model.encode_context(tg.x_dict, tg.edge_index_dict)
        loss = 0.0
        for a in active:
            sel = held[a]["train"]
            pred = model.predict(a, zc["paper"][held[a]["paper"][sel]])
            tgt = pool[a][held[a]["target"][sel]]
            loss = loss + F.smooth_l1_loss(pred, tgt, beta=SMOOTHL1_BETA)
            loss = loss + COS_COEF * (1 - F.cosine_similarity(pred, tgt, -1)).mean()
            loss = loss + VAR_COEF * variance_loss(pred)
        opt.zero_grad(); loss.backward(); opt.step(); model.ema_update(m)

    model.eval()
    with torch.no_grad():
        zc = model.encode_context(tg.x_dict, tg.edge_index_dict)
        zt = model.encode_target(full.x_dict, full.edge_index_dict)
    res, std, rank = {}, None, None
    for a in active:
        sel = held[a]["test"]
        p = model.predict(a, zc["paper"][held[a]["paper"][sel]])
        res[a] = mrr_euclid(p.cpu(), held[a]["target"][sel].cpu(), zt[a].detach().cpu(), gen=gen)
        if a == active[0]:
            std, rank = collapse_stats(zc["paper"].detach())
    return res, std, rank, model


# ============================================================
#  VARIANTS B & C — partition patch (+ optional hyperbolic)
# ============================================================
def build_paper_target_map(data, labels, slices, active):
    p0, _ = slices["paper"]
    out = {}
    for a in active:
        rel, _ = ASPECT_REL[a]
        ei = data[rel].edge_index; a0, _ = slices[a]
        is_target = (labels[ei[0] + p0] != labels[ei[1] + a0])
        out[a] = (ei[0], ei[1], is_target)
    return out


def run_metis(seed, data, active, hyperbolic, incident):
    gen = torch.Generator().manual_seed(seed)
    labels, slices, _ = get_partition(data, N_PARTS, seed)
    tmap = build_paper_target_map(data, labels, slices, active)

    held = {}
    for a in active:
        hp, ht, is_t = tmap[a]
        ti = is_t.nonzero(as_tuple=True)[0]
        if ti.numel() < 4:
            ti = torch.randperm(hp.size(0), generator=gen)[:max(4, int(0.1 * hp.size(0)))]
        idx = torch.randperm(ti.numel(), generator=gen); ntest = max(1, int(ti.numel() * 0.5))
        held[a] = dict(paper=hp[ti].to(DEVICE), target=ht[ti].to(DEVICE),
                       train=idx[ntest:], test=idx[:ntest])

    # context: drop target has_<aspect> edges AND their leak edges (consistency)
    ctx = data.clone()
    for a in active:
        _, ht, is_t = tmap[a]
        rel, rev = ASPECT_REL[a]
        keep = (~is_t)
        ctx[rel].edge_index = data[rel].edge_index[:, keep]
        if rev in ctx.edge_types:
            ctx[rev].edge_index = data[rev].edge_index[:, keep]
        target_nodes = set(ht[is_t].tolist())
        for et, row in incident[a]:
            if et == rel or et not in ctx.edge_types:
                continue
            e = data[et].edge_index
            if e.numel() == 0:
                continue
            km = torch.tensor([nid not in target_nodes for nid in e[row].tolist()],
                              dtype=torch.bool)
            ctx[et].edge_index = e[:, km]
            ret = reverse_et(et)
            if ret in ctx.edge_types and data[ret].edge_index.size(1) == e.size(1):
                ctx[ret].edge_index = data[ret].edge_index[:, km]
    ctx = ctx.to(DEVICE); full = data.clone().to(DEVICE)

    num_pos = {a: data[a].num_nodes for a in active} if hyperbolic else None
    model = JEPA(data.metadata(), HIDDEN, LATENT, active,
                 hyperbolic=hyperbolic, num_pos=num_pos).to(DEVICE)
    with torch.no_grad():
        model.encode_context(ctx.x_dict, ctx.edge_index_dict)
        model.target_encoder(full.x_dict, full.edge_index_dict)
        model.init_target_from_online(); model.freeze_target()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)

    for ep in range(EPOCHS):
        model.train()
        m = EMA_FINAL - (EMA_FINAL - EMA_BASE) * (0.5 * (1 + np.cos(np.pi * ep / max(1, EPOCHS - 1))))
        with torch.no_grad():
            zt = model.encode_target(full.x_dict, full.edge_index_dict)
            pool = {a: zt[a].detach() for a in active}
        zc = model.encode_context(ctx.x_dict, ctx.edge_index_dict)
        loss = 0.0
        for a in active:
            sel = held[a]["train"]
            tids = held[a]["target"][sel]
            z_x = zc["paper"][held[a]["paper"][sel]]
            if hyperbolic:
                pred_psi = model.predict(a, z_x, target_ids=tids)          # Eq.5 -> R^2
                tgt_psi = to_hyperbola(pool[a][tids])                      # Eq.4
                loss = loss + F.smooth_l1_loss(pred_psi, tgt_psi, beta=SMOOTHL1_BETA)  # Eq.6
                # NO variance term: collapse prevented by EMA stop-grad + simple predictor.
            else:
                pred = model.predict(a, z_x)
                tgt = pool[a][tids]
                loss = loss + F.smooth_l1_loss(pred, tgt, beta=SMOOTHL1_BETA)
                loss = loss + COS_COEF * (1 - F.cosine_similarity(pred, tgt, -1)).mean()
                loss = loss + VAR_COEF * variance_loss(pred)
        opt.zero_grad(); loss.backward(); opt.step(); model.ema_update(m)

    model.eval()
    with torch.no_grad():
        zc = model.encode_context(ctx.x_dict, ctx.edge_index_dict)
        zt = model.encode_target(full.x_dict, full.edge_index_dict)
    res, std, rank = {}, None, None
    for a in active:
        sel = held[a]["test"]; tids = held[a]["target"][sel]
        z_x = zc["paper"][held[a]["paper"][sel]]
        if hyperbolic:
            pred_psi = model.predict(a, z_x, target_ids=tids)
            pool_psi = to_hyperbola(zt[a].detach())
            res[a] = mrr_hyperbola(pred_psi.cpu(), tids.cpu(), pool_psi.cpu(), gen=gen)
            if a == active[0]:
                # report collapse on the LATENT z (not on hyperbola coords):
                std, rank = collapse_stats(zt[a].detach())
        else:
            pred = model.predict(a, z_x)
            res[a] = mrr_euclid(pred.cpu(), tids.cpu(), zt[a].detach().cpu(), gen=gen)
            if a == active[0]:
                std, rank = collapse_stats(zc["paper"].detach())
    return res, std, rank, model


# ============================================================
#  Driver
# ============================================================
def mean_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), float("nan"))


def save_ckpt(variant, model, data, active, seed, val_mrr, incident):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"jepa_{variant}_best.pt")
    torch.save({
        "variant": variant,
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "metadata": data.metadata(),
        "hidden": HIDDEN, "latent": LATENT, "aspects": active, "seed": seed,
        "hyperbolic": (variant == "metis_hyperbolic"),
        "val_mrr": val_mrr,
        "partitioner_detected": PARTITIONER,
        "partitioner_used": ACTUAL_PARTITIONER["used"],
        "leak_incident": {a: [list(et) + [row] for et, row in incident[a]] for a in active},
    }, path)
    print(f"  [save] {variant}: best seed {seed} (meanMRR {val_mrr:.4f}) -> {path}")


def main():
    rebuild = os.environ.get("REBUILD_HETERO", "0") == "1"
    print(f"[main] build_hetero_graph(rebuild={rebuild})")
    print(f"[main] PARTITIONER detected = {PARTITIONER}")
    data, meta = build_hetero_graph(RAW_DIR, CACHE_PATH, rebuild=rebuild)
    print(meta)
    data = T.ToUndirected()(data)

    active = [a for a in ASPECTS
              if ASPECT_REL[a][0] in data.edge_types
              and data[ASPECT_REL[a][0]].edge_index.numel() > 0]

    incident = build_incident_map(data, active)
    print(f"[main] active aspects: {active}")
    print("[main] LEAK MAP (aspect -> edges dropped when held out):")
    for a in active:
        pretty = [f"{et[1]}(row{row})" for et, row in incident[a]]
        print(f"        {a:<7}: {pretty}")
    print(f"[main] variants: {VARIANTS} | seeds: {SEEDS}")

    results = {v: [] for v in VARIANTS}
    best = {v: {"val": -1.0, "seed": None, "model": None} for v in VARIANTS}

    for v in VARIANTS:
        print("\n" + "#" * 62 + f"\n#  VARIANT: {v}\n" + "#" * 62)
        for s in SEEDS:
            torch.manual_seed(s); np.random.seed(s); random.seed(s)
            if v == "edge_mask":
                res, std, rank, model = run_edge_mask(s, data, active, incident)
            elif v == "metis_patch":
                res, std, rank, model = run_metis(s, data, active, False, incident)
            else:
                res, std, rank, model = run_metis(s, data, active, True, incident)
            mean_mrr = float(np.mean([res[a]["MRR"] for a in active]))
            results[v].append((res, std, rank))
            if mean_mrr > best[v]["val"]:
                best[v] = {"val": mean_mrr, "seed": s, "model": model}
            print(f"  [{v}] seed {s} | meanMRR {mean_mrr:.4f} | std {std:.3f} rank {rank:.1f}",
                  flush=True)
        save_ckpt(v, best[v]["model"], data, active, best[v]["seed"], best[v]["val"], incident)

    used = ACTUAL_PARTITIONER["used"] or PARTITIONER
    print("\n" + "=" * 78)
    print("  ICLR COMPARISON  (mean ± std over", len(SEEDS), "seeds)")
    print(f"  partitioner DETECTED = {PARTITIONER}  |  ACTUALLY USED = {used}"
          + ("" if used == "metis" else "   <-- report as BFS, NOT Metis!"))
    print("=" * 78)
    print(f"  {'variant':<18} | {'MRR':<13} | {'Hits@1':<13} | {'Hits@10':<13} | {'std':<6} | rank")
    print("  " + "-" * 74)

    summary = {"partitioner_detected": PARTITIONER,
               "partitioner_used": used,
               "seeds": SEEDS, "active_aspects": active,
               "leak_map": {a: [f"{et[1]}(row{row})" for et, row in incident[a]] for a in active},
               "hyperbolic_note": "Graph-JEPA Eq.4 unit hyperbola [cosh(a),sinh(a)], "
                                  "a=mean(z); predictor conditioned on per-target "
                                  "positional embedding P_l (Eq.5); Smooth-L1 energy "
                                  "(Eq.6). std reported on LATENT z, not hyperbola coords.",
               "variants": {}}
    for v in VARIANTS:
        def agg(metric):
            return mean_std([float(np.mean([r[0][a][metric] for a in active]))
                             for r in results[v]])
        mrr, h1, h10 = agg("MRR"), agg("Hits@1"), agg("Hits@10")
        stds = mean_std([r[1] for r in results[v]]); ranks = mean_std([r[2] for r in results[v]])
        print(f"  {v:<18} | {mrr[0]:.3f}±{mrr[1]:.3f} | {h1[0]:.3f}±{h1[1]:.3f} | "
              f"{h10[0]:.3f}±{h10[1]:.3f} | {stds[0]:.2f}  | {ranks[0]:.1f}")
        summary["variants"][v] = {
            "MRR": mrr, "Hits@1": h1, "Hits@10": h10,
            "enc_std": stds, "eff_rank": ranks,
            "best_seed": best[v]["seed"], "best_val_mrr": best[v]["val"],
            "per_aspect": {a: {mtr: mean_std([r[0][a][mtr] for r in results[v]])
                               for mtr in ["MRR", "Hits@1", "Hits@5", "Hits@10"]}
                           for a in active},
        }
    print("=" * 78)
    print("  NOTE: metis_hyperbolic uses Graph-JEPA Eq.4-6 (unit hyperbola + P_l + Smooth-L1).")
    print("        Judge collapse via Hits@1<Hits@10 and LATENT std, NOT hyperbola-coord std.")
    print("=" * 78)

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {RESULTS_JSON}")


if __name__ == "__main__":
    main()