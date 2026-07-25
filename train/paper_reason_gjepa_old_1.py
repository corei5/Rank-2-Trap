"""
paper_reason_gjepa.py — Graph-JEPA hyperbola collapse: CAUSE, CONTROLS,
                        LORENTZ ALTERNATIVE, and the POOLING SWEEP (v5.2).
============================================================================
graph = PAPER ; patches = ASPECT subgraphs ; within-paper MASKED PREDICTION.
Consumes cached RWSE from build_rwse.py. Faithful Graph-JEPA core: EMA target,
stop-grad, GIN-in-to_hetero, RWSE PE, 3-layer predictor, Smooth-L1 on hyperbola.

============================================================================
THE FULL STORY (variants x pooling modes)
============================================================================
DIAGNOSIS (proven): raw GIN NODE embeddings have eff_rank ~18/128 (healthy),
  but MEAN-pooling them to patch vectors collapses eff_rank -> ~2/128. Retrieval
  then fails (MRR ~chance) for EVERY objective (faithful / VICReg / per-dim
  hyperbola / Lorentz), because they all act AFTER the rank-destroying pool.
  => the collapse is in the POOLING, not the encoder or the objective.

POOLING SWEEP (the ablation):
  [MOD6a] attn      : learned attention pooling (a convex combination -> still
                      concentrates like a mean; empirically does NOT recover rank).
  [MOD6b] multistat : concat [mean || max || std] -> Linear -> LayerNorm.
  We sweep EVERY variant x EVERY pooling mode and report the 3-level rank probe
  (node -> patch -> pool) + MRR for each.

PATCH v5.1: robust eff_rank (float64 + ridge + SVD fallback) fixes eigvalsh err 129.
PATCH v5.2: MultiStatPool + LayerNorm fixes degenerate output (pm_rk=NaN) and the
            spurious MRR=1.000 self-match; NaN-safe rank aggregation.

Run:  python -m train.paper_reason_gjepa   (after build_rwse.py)
"""

import os, json, random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, Linear as PygLinear, to_hetero
from torch_geometric.utils import scatter
import torch_geometric.transforms as T

from core.data_utils.paper_graph import build_hetero_graph, ASPECTS


# ─────────────────────────────────────────────────────────────
RAW_DIR    = "/nfs/home/rabbyg/JEPA/Graph-JEPA/dataset/papers/raw"
CACHE_PATH = "/nfs/home/rabbyg/JEPA/Graph-JEPA/dataset/papers/processed/hetero_graphA.pt"
RWSE_CACHE = "/nfs/home/rabbyg/JEPA/Graph-JEPA/dataset/papers/processed/rwse_pe.pt"
CKPT_DIR   = "/nfs/home/rabbyg/JEPA/Graph-JEPA/checkpoints/reasoning"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN, LATENT = 128, 128
EPOCHS, LR, WD = 100, 1e-3, 1e-5
SEEDS          = [0, 1, 2, 3, 4]
EMA_BASE, EMA_FINAL = 0.996, 1.0
SMOOTHL1_BETA  = 1.0

VIC_GAMMA, VIC_EPS = 1.0, 1e-4
LORENTZ_C, LORENTZ_EPS = 1.0, 1e-6

# ---- OBJECTIVE variants (v1-v4) ----
VARIANT_CFG = {
    "metis_hyperbolic":            dict(loss_mode="mean",    use_vic=False, inv=1.00, std=0.0,  cov=0.0),
    "metis_hyperbolic_vic":        dict(loss_mode="mean",    use_vic=True,  inv=0.10, std=25.0, cov=1.0),
    "metis_hyperbolic_vic_strong": dict(loss_mode="mean",    use_vic=True,  inv=0.01, std=50.0, cov=1.0),
    "metis_hyperbolic_full":       dict(loss_mode="full",    use_vic=False, inv=1.00, std=0.0,  cov=0.0),
    "metis_lorentz":               dict(loss_mode="lorentz", use_vic=False, inv=1.00, std=0.0,  cov=0.0),
}
VARIANTS = list(VARIANT_CFG.keys())

# ---- POOLING modes. "mean" = baseline; "attn"/"multistat" = candidates. ----
POOL_MODES = ["mean", "attn", "multistat"]

RESULTS_JSON   = os.path.join(CKPT_DIR, "gjepa_ablation_results.json")
COLLAPSE_STD_THRESH  = 0.30
COLLAPSE_RANK_THRESH = 5.0
# ─────────────────────────────────────────────────────────────

ASPECT_REL = {a: ("paper", f"has_{a}", a) for a in ASPECTS}


# ============================================================
#  PATCH INDEX / PRESENCE / LABEL
# ============================================================
def build_patch_index(data, active):
    idx = {}
    for a in active:
        rel = ASPECT_REL[a]
        if rel not in data.edge_types:
            idx[a] = (torch.empty(0, dtype=torch.long),
                      torch.empty(0, dtype=torch.long)); continue
        ei = data[rel].edge_index
        idx[a] = (ei[1], ei[0])
    return idx


def aspect_presence(data, patch_idx, active):
    P = data["paper"].num_nodes
    pres = torch.zeros(P, len(active), dtype=torch.bool)
    for col, a in enumerate(active):
        _, owner = patch_idx[a]
        if owner.numel() > 0:
            pres[owner.unique(), col] = True
    return pres


def build_reasoning_label(data):
    P = data["paper"].num_nodes
    sup = torch.zeros(P); cha = torch.zeros(P)
    hc = data[("paper", "has_claim", "claim")].edge_index
    claim_owner = torch.full((data["claim"].num_nodes,), -1, dtype=torch.long)
    claim_owner[hc[1]] = hc[0]
    for et, tgt in [(("claim", "supported_by", "evidence"), sup),
                    (("claim", "challenged_by", "evidence"), cha)]:
        if et not in data.edge_types:
            continue
        ei = data[et].edge_index
        owners = claim_owner[ei[0]]; valid = owners >= 0
        tgt.index_add_(0, owners[valid], torch.ones(int(valid.sum())))
    label = torch.full((P,), -1, dtype=torch.long)
    has_ev = (sup + cha) > 0
    label[has_ev] = (cha[has_ev] >= sup[has_ev]).long()
    return label


# ============================================================
#  GIN ENCODER
# ============================================================
class GINEncoder(torch.nn.Module):
    def __init__(self, hidden, out):
        super().__init__()
        assert hidden == out, "GIN residual requires hidden == out"
        self.proj = PygLinear(-1, hidden)
        nn1 = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                                  torch.nn.Linear(hidden, hidden))
        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden, out), torch.nn.GELU(),
                                  torch.nn.Linear(out, out))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index):
        x = F.gelu(self.proj(x))
        x = F.gelu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


# ============================================================
#  TARGET PROJECTIONS  (mean / full hyperbola / Lorentz)
# ============================================================
def to_hyperbola(z):
    a = z.mean(-1, keepdim=True)
    return torch.cat([torch.cosh(a), torch.sinh(a)], -1)


def to_hyperbola_full(z):
    return torch.stack([torch.cosh(z), torch.sinh(z)], dim=-1).flatten(-2)


def lorentz_expmap0(v, c=LORENTZ_C, eps=LORENTZ_EPS):
    sqrt_c = c ** 0.5
    vnorm = torch.clamp(v.norm(dim=-1, keepdim=True), min=eps)
    x0 = torch.cosh(sqrt_c * vnorm) / sqrt_c
    xs = torch.sinh(sqrt_c * vnorm) * v / (sqrt_c * vnorm)
    return torch.cat([x0, xs], dim=-1)


def lorentz_inner(x, y):
    return -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(-1, keepdim=True)


def lorentz_sqdist(x, y, c=LORENTZ_C, eps=LORENTZ_EPS):
    prod = torch.clamp(-c * lorentz_inner(x, y).squeeze(-1), min=1.0 + eps)
    d = torch.acosh(prod) / (c ** 0.5)
    return d * d


def project_target(z, mode):
    if mode == "full":    return to_hyperbola_full(z)
    if mode == "lorentz": return lorentz_expmap0(z)
    return to_hyperbola(z)


def mlp3(inp, out):
    return torch.nn.Sequential(
        torch.nn.Linear(inp, inp), torch.nn.GELU(),
        torch.nn.Linear(inp, inp), torch.nn.GELU(),
        torch.nn.Linear(inp, out))


# ============================================================
#  POOLING  — mean (baseline) | attn | multistat
# ============================================================
class AttnPool(torch.nn.Module):
    """[MOD6a] Learned attention pooling: score each member node, softmax within
    its owner patch, weighted sum. NOTE: a convex combination -> still an average,
    so empirically it does NOT escape the mean's rank concentration."""
    def __init__(self, dim):
        super().__init__()
        self.score = torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.GELU(),
                                          torch.nn.Linear(dim, 1))

    def forward(self, z, member, owner, P):
        if member.numel() == 0:
            return torch.zeros(P, z.size(1), device=z.device)
        h = z[member]
        s = self.score(h).squeeze(-1)
        s = s - scatter(s, owner, dim=0, dim_size=P, reduce="max")[owner]  # stable softmax
        e = torch.exp(s)
        denom = scatter(e, owner, dim=0, dim_size=P, reduce="sum").clamp_min(1e-9)
        w = (e / denom[owner]).unsqueeze(-1)
        return scatter(h * w, owner, dim=0, dim_size=P, reduce="sum")


class MultiStatPool(torch.nn.Module):
    """[MOD6b, v5.2 FIXED] concat [mean || max || std] -> Linear -> LayerNorm.
    LayerNorm + std clamp prevent the degenerate all-equal output that produced
    NaN eff_rank and a spurious MRR=1.000 self-match in the earlier version."""
    def __init__(self, dim):
        super().__init__()
        self.merge = torch.nn.Linear(3 * dim, dim)
        self.norm  = torch.nn.LayerNorm(dim)

    def forward(self, z, member, owner, P):
        D = z.size(1)
        if member.numel() == 0:
            return torch.zeros(P, D, device=z.device)
        h = z[member]
        mean = scatter(h, owner, dim=0, dim_size=P, reduce="mean")
        mx   = scatter(h, owner, dim=0, dim_size=P, reduce="max")
        msq  = scatter(h * h, owner, dim=0, dim_size=P, reduce="mean")
        std  = (msq - mean * mean).clamp_min(1e-6).sqrt()          # v5.2: clamp>0
        out  = self.merge(torch.cat([mean, mx, std], dim=-1))
        return self.norm(out)                                      # v5.2: LayerNorm


# ============================================================
#  VICReg
# ============================================================
def vicreg_terms(z, gamma=VIC_GAMMA, eps=VIC_EPS):
    N, D = z.shape
    if N < 2:
        zero = z.sum() * 0.0
        return zero, zero
    zc = z - z.mean(0, keepdim=True)
    std = torch.sqrt(zc.var(0, unbiased=True) + eps)
    var_term = torch.mean(F.relu(gamma - std))
    cov = (zc.T @ zc) / (N - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_term, (off.pow(2).sum()) / D


# ============================================================
#  COLLAPSE DIAGNOSTICS  (v5.1 robust; fixes eigvalsh err 129)
# ============================================================
@torch.no_grad()
def collapse_stats(z, max_n=8000):
    if z.size(0) > max_n:
        z = z[torch.randperm(z.size(0), device=z.device)[:max_n]]
    N, D = z.shape
    if N < 2:
        return {"latent_std": float("nan"), "offdiag_cov": float("nan"),
                "eff_rank": float("nan")}
    zc = z - z.mean(0, keepdim=True)
    std = zc.std(0, unbiased=True)
    zc64 = zc.double()
    cov = (zc64.T @ zc64) / (N - 1)
    off = cov - torch.diag(torch.diag(cov))
    ridge = 1e-6 * (torch.diag(cov).mean().clamp_min(1e-12))
    cov_r = (cov + ridge * torch.eye(D, dtype=cov.dtype, device=cov.device)).cpu()
    eig = None
    try:
        eig = torch.linalg.eigvalsh(cov_r)
    except Exception:
        try:
            eig = torch.linalg.svdvals(cov_r)
        except Exception:
            eig = None
    if eig is None:
        eff_rank = float("nan")
    else:
        eig = eig.clamp_min(0)
        p = eig / eig.sum().clamp_min(1e-12)
        entropy = -(p * (p + 1e-12).log()).sum()
        eff_rank = float(torch.exp(entropy))
    return {"latent_std": float(std.mean()),
            "offdiag_cov": float(off.abs().mean()),
            "eff_rank": eff_rank}


@torch.no_grad()
def eff_rank_only(z, max_n=8000):
    try:
        return collapse_stats(z, max_n)["eff_rank"]
    except Exception:
        return float("nan")


# ============================================================
#  MODEL  (pooling selectable per run)
# ============================================================
class GraphJEPA(torch.nn.Module):
    def __init__(self, metadata, hidden, latent, active, pe_dim, pool_mode):
        super().__init__()
        self.active = list(active); self.pool_mode = pool_mode
        self.node_enc = to_hetero(GINEncoder(hidden, latent), metadata, aggr="sum")
        self.node_tgt = to_hetero(GINEncoder(hidden, latent), metadata, aggr="sum")
        enc = torch.nn.TransformerEncoderLayer(latent, 4, latent * 2,
                                               batch_first=True, activation="gelu")
        self.ctx_mixer = torch.nn.TransformerEncoder(enc, 2)
        self.pe_proj   = torch.nn.Linear(pe_dim, latent)
        self.predictor = mlp3(latent, latent)
        if pool_mode == "attn":
            self.pool_ctx = AttnPool(latent); self.pool_tgt = AttnPool(latent)
        elif pool_mode == "multistat":
            self.pool_ctx = MultiStatPool(latent); self.pool_tgt = MultiStatPool(latent)
        else:
            self.pool_ctx = self.pool_tgt = None

    def encode_nodes(self, x, ei):     return self.node_enc(x, ei)
    @torch.no_grad()
    def encode_nodes_tgt(self, x, ei): return self.node_tgt(x, ei)

    def patch_embed(self, zdict, patch_idx, a, P, which):
        z = zdict[a]; member, owner = patch_idx[a]
        if member.numel() == 0:
            return torch.zeros(P, z.size(1), device=z.device)
        member = member.to(z.device); owner = owner.to(z.device)
        if self.pool_mode == "mean":
            return scatter(z[member], owner, dim=0, dim_size=P, reduce="mean")
        pool = self.pool_ctx if which == "ctx" else self.pool_tgt
        return pool(z, member, owner, P)

    @torch.no_grad()
    def init_target(self):
        self.node_tgt.load_state_dict(self.node_enc.state_dict())
        if self.pool_mode in ("attn", "multistat"):
            self.pool_tgt.load_state_dict(self.pool_ctx.state_dict())

    @torch.no_grad()
    def freeze_target(self):
        for p in self.node_tgt.parameters(): p.requires_grad_(False)
        if self.pool_mode in ("attn", "multistat"):
            for p in self.pool_tgt.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def ema(self, m):
        for pq, pk in zip(self.node_enc.parameters(), self.node_tgt.parameters()):
            pk.data.mul_(m).add_((1 - m) * pq.detach().data)
        if self.pool_mode in ("attn", "multistat"):
            for pq, pk in zip(self.pool_ctx.parameters(), self.pool_tgt.parameters()):
                pk.data.mul_(m).add_((1 - m) * pq.detach().data)


# ============================================================
#  TRAIN
# ============================================================
def train_gjepa(seed, data, active, patch_idx, pres, rwse, maskable, pe_dim,
                cfg, pool_mode, verbose_diag=False):
    torch.manual_seed(seed)
    full = data.to(DEVICE)
    P = full["paper"].num_nodes; A = len(active)
    pe = {a: rwse[a].to(DEVICE) for a in active}
    pres_d = pres.to(DEVICE); maskable_idx = maskable.to(DEVICE)
    model = GraphJEPA(data.metadata(), HIDDEN, LATENT, active, pe_dim, pool_mode).to(DEVICE)

    with torch.no_grad():
        model.encode_nodes(full.x_dict, full.edge_index_dict)
        model.encode_nodes_tgt(full.x_dict, full.edge_index_dict)
    model.init_target(); model.freeze_target()

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=LR, weight_decay=WD)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    mode = cfg["loss_mode"]; diag_trace = []

    for ep in range(EPOCHS):
        model.train()
        m = EMA_FINAL - (EMA_FINAL - EMA_BASE) * (0.5 * (1 + np.cos(np.pi * ep / max(1, EPOCHS - 1))))
        with torch.no_grad():
            zt = model.encode_nodes_tgt(full.x_dict, full.edge_index_dict)
        zc = model.encode_nodes(full.x_dict, full.edge_index_dict)

        ctx_emb = torch.stack([model.patch_embed(zc, patch_idx, a, P, "ctx") for a in active], 1)
        with torch.no_grad():
            tgt_emb = torch.stack([model.patch_embed({k: zt[k].detach() for k in active},
                                                     patch_idx, a, P, "tgt") for a in active], 1)

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

        if mode == "lorentz":
            inv_loss = lorentz_sqdist(lorentz_expmap0(pred_latent),
                                      lorentz_expmap0(tgt_patch)).mean()
        else:
            inv_loss = F.smooth_l1_loss(project_target(pred_latent, mode),
                                        project_target(tgt_patch,   mode), beta=SMOOTHL1_BETA)
        loss = cfg["inv"] * inv_loss
        if cfg["use_vic"]:
            v_e, c_e = vicreg_terms(ctx_emb[maskable_idx][pres_m])
            loss = loss + cfg["std"] * v_e + cfg["cov"] * c_e

        opt.zero_grad(); loss.backward(); opt.step(); model.ema(m)

    model.eval()
    with torch.no_grad():
        zt = model.encode_nodes_tgt(full.x_dict, full.edge_index_dict)
        patch_repr = torch.stack(
            [model.patch_embed({k: zt[k] for k in active}, patch_idx, a, P, "tgt") for a in active], 1)
        try:
            node_ranks = {a: eff_rank_only(zt[a]) for a in active}
            pm_flat = patch_repr[maskable_idx][pres_d[maskable_idx]]
            # v5.2: NaN-safe aggregation (no np.nanmean warning / propagation)
            vals = [v for v in node_ranks.values() if not np.isnan(v)]
            upstream = {"node_eff_rank_mean": float(np.mean(vals)) if vals else float("nan"),
                        "node_eff_rank_per_aspect": {a: float(node_ranks[a]) for a in active},
                        "patchmean_eff_rank": float(eff_rank_only(pm_flat))}
        except Exception as e:
            print(f"    [warn] rank probe failed: {e}")
            upstream = {"node_eff_rank_mean": float("nan"),
                        "node_eff_rank_per_aspect": {a: float("nan") for a in active},
                        "patchmean_eff_rank": float("nan")}
    return model, patch_repr, pe, pres_d, maskable_idx, diag_trace, upstream, mode


# ============================================================
#  A1 PROBE
# ============================================================
def eval_probe(patch_repr, pres_d, active, label, seed):
    g = torch.Generator().manual_seed(seed)
    P, A, L = patch_repr.shape
    w = pres_d.float().unsqueeze(-1)
    reps = ((patch_repr * w).sum(1) / w.sum(1).clamp_min(1.0)).cpu()
    mask = label >= 0
    X, y = reps[mask], label[mask]
    idx = torch.randperm(X.size(0), generator=g); ntr = int(0.7 * X.size(0))
    Xtr, ytr, Xte, yte = X[idx[:ntr]], y[idx[:ntr]], X[idx[ntr:]], y[idx[ntr:]]
    clf = torch.nn.Linear(L, int(y.max()) + 1)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(300):
        opt.zero_grad(); F.cross_entropy(clf(Xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        pred = clf(Xte).argmax(1)
        acc = (pred == yte).float().mean().item()
        baccs = [(pred[yte==c]==yte[yte==c]).float().mean().item()
                 for c in range(int(y.max())+1) if (yte==c).sum()>0]
        bacc = float(np.mean(baccs)) if baccs else acc
        maj = float(max((yte==c).float().mean().item() for c in range(int(y.max())+1)))
    return {"acc": acc, "balanced_acc": bacc, "majority_baseline": maj}


# ============================================================
#  A2 RETRIEVAL  (metric by loss-mode; pooling baked into patch_repr)
#  NOTE: query = predictor(context WITHOUT the target aspect); candidate pool =
#  frozen-target patches. The true target is a candidate but is NOT fed to the
#  query path, so there is no identity self-match leak.
# ============================================================
@torch.no_grad()
def eval_patchret(model, patch_repr, pe, pres_d, active, maskable_idx, seed, mode,
                  neg=1000, query_bs=2048):
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 777)
    P, A, L = patch_repr.shape; Pm = maskable_idx.size(0)
    if Pm < 2:
        return {"MRR": float("nan"), "Hits@1": float("nan"),
                "Hits@10": float("nan"), **collapse_stats(patch_repr.reshape(-1, L))}
    pres_m = pres_d[maskable_idx]
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
    if mode == "lorentz":
        pred_h = lorentz_expmap0(pred); pool_h = lorentz_expmap0(pool)
    ranks = torch.empty(Pm, device=DEVICE)
    for start in range(0, Pm, query_bs):
        end = min(start + query_bs, Pm); b = end - start
        true_local = torch.arange(start, end, device=DEVICE)
        if Pm > neg + 1:
            r = torch.randint(0, Pm, (b, neg), generator=gen, device=DEVICE)
            cand = torch.cat([true_local.view(-1, 1), r], 1)
        else:
            cand = torch.arange(Pm, device=DEVICE).unsqueeze(0).expand(b, Pm)
        if mode == "lorentz":
            q = pred_h[start:end]; cp = pool_h[cand]
            d = lorentz_sqdist(q.unsqueeze(1).expand_as(cp), cp)
        else:
            q = pred[start:end]; cp = pool[cand]
            d = F.smooth_l1_loss(q.unsqueeze(1).expand_as(cp), cp,
                                 beta=SMOOTHL1_BETA, reduction="none").mean(-1)
        ranks[start:end] = (d < d[:, :1]).sum(1).float() + 1
    return {"MRR": float((1/ranks).mean()),
            "Hits@1": float((ranks<=1).float().mean()),
            "Hits@10": float((ranks<=10).float().mean()), **cstats}


# ============================================================
#  DRIVER
# ============================================================
def mean_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), float("nan"))


def verdict(lat_std, eff_rank):
    if np.isnan(lat_std) or np.isnan(eff_rank): return "n/a"
    return "COLLAPSED" if (lat_std < COLLAPSE_STD_THRESH or eff_rank < COLLAPSE_RANK_THRESH) else "RECOVERED"


def run_cell(name, pool_mode, data, active, patch_idx, pres, label, rwse, maskable, pe_dim):
    cfg = VARIANT_CFG[name]
    print(f"\n### {name}  x  pool={pool_mode}")
    A1, A2, UPS = [], [], []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        model, patch_repr, pe, pres_d, maskable_idx, _, ups, mode = train_gjepa(
            s, data, active, patch_idx, pres, rwse, maskable, pe_dim, cfg=cfg, pool_mode=pool_mode)
        r1 = eval_probe(patch_repr, pres_d, active, label, s)
        r2 = eval_patchret(model, patch_repr, pe, pres_d, active, maskable_idx, s, mode)
        A1.append(r1); A2.append(r2); UPS.append(ups)
        print(f"  seed {s} | A1 {r1['acc']:.3f} | MRR {r2['MRR']:.3f} H@10 {r2['Hits@10']:.3f} "
              f"| pool_rk {r2['eff_rank']:.1f} node_rk {ups['node_eff_rank_mean']:.1f} "
              f"pm_rk {ups['patchmean_eff_rank']:.1f}", flush=True)
    acc  = mean_std([r["acc"] for r in A1]); mrr = mean_std([r["MRR"] for r in A2])
    h10  = mean_std([r["Hits@10"] for r in A2]); h1 = mean_std([r["Hits@1"] for r in A2])
    lstd = mean_std([r["latent_std"] for r in A2]); erank = mean_std([r["eff_rank"] for r in A2])
    nrank= mean_std([u["node_eff_rank_mean"] for u in UPS])
    pmrank=mean_std([u["patchmean_eff_rank"] for u in UPS])
    return {"variant": name, "pool_mode": pool_mode,
            "A1_acc": acc, "MRR": mrr, "Hits@1": h1, "Hits@10": h10,
            "pool_eff_rank": erank, "node_eff_rank": nrank, "patchmean_eff_rank": pmrank,
            "verdict": verdict(lstd[0], erank[0])}


def main():
    if not os.path.exists(RWSE_CACHE):
        raise FileNotFoundError(f"RWSE cache missing: {RWSE_CACHE}\nRun build_rwse first.")
    blob = torch.load(RWSE_CACHE, weights_only=False)
    rwse   = blob["patch_rwse"]; pe_dim = blob["rwse_steps"]
    print(f"[gjepa] loaded RWSE cache | steps={pe_dim} | verdict={blob['report']}")
    data, _ = build_hetero_graph(RAW_DIR, CACHE_PATH, rebuild=False)
    data = T.ToUndirected()(data)
    active = blob["active"]
    patch_idx = build_patch_index(data, active)
    pres      = aspect_presence(data, patch_idx, active)
    label     = build_reasoning_label(data)
    ncnt = pres.sum(1); maskable = torch.nonzero(ncnt >= 2, as_tuple=False).squeeze(1)
    P = data["paper"].num_nodes
    print(f"[gjepa] active={active} | MASKABLE={maskable.numel()} ({100*maskable.numel()/max(1,P):.1f}%)")
    print(f"[gjepa] label supported={int((label==0).sum())} challenged/mixed={int((label==1).sum())}")

    cells = []
    for name in VARIANTS:
        for pm in POOL_MODES:
            try:
                cells.append(run_cell(name, pm, data, active, patch_idx, pres, label,
                                      rwse, maskable, pe_dim))
            except Exception as e:
                print(f"  [ERROR] cell {name} x {pm} failed: {e}")

    # ---- MASTER TABLE : variant x pooling ----
    print("\n" + "=" * 112)
    print("  VARIANT x POOLING SWEEP — cause (mean) & candidate pools (attn/multistat)  "
          f"(mean ± std / {len(SEEDS)} seeds)")
    print("=" * 112)
    print(f"  {'variant':<26}| {'pool':<10}| {'A1 acc':<11}| {'MRR':<11}| {'H@10':<7}| "
          f"{'pool_rk':<8}| {'node_rk':<8}| {'pm_rk':<7}| verdict")
    print("  " + "-" * 108)
    for c in cells:
        print(f"  {c['variant']:<26}| {c['pool_mode']:<10}| "
              f"{c['A1_acc'][0]:.3f}±{c['A1_acc'][1]:.2f}| "
              f"{c['MRR'][0]:.3f}±{c['MRR'][1]:.2f}| {c['Hits@10'][0]:<7.3f}| "
              f"{c['pool_eff_rank'][0]:.1f}/{LATENT:<3}| {c['node_eff_rank'][0]:.1f}/{LATENT:<3}| "
              f"{c['patchmean_eff_rank'][0]:.1f}/{LATENT:<2}| {c['verdict']}")
    print("=" * 112)
    print("  READING: pm_rk (patch rank) is the diagnostic. pool='mean' -> pm_rk~2 (collapse).")
    print("  A real fix must RAISE pm_rk WELL above 2 AND lift MRR while KEEPING A1 high (~0.85).")
    print("  A cell with A1~0.57 (majority) or MRR=1.000 is a degenerate/leaking artifact, NOT a fix.")
    print("=" * 112)

    os.makedirs(CKPT_DIR, exist_ok=True)
    json.dump({
        "method": "Graph-JEPA hyperbola collapse: cause, controls, Lorentz alternative, and a "
                  "pooling sweep. {objective variants} x {mean, attn, multistat} pooling.",
        "cells": cells, "variants": VARIANTS, "pool_modes": POOL_MODES,
        "active_aspects": active, "seeds": SEEDS,
        "maskable_papers": int(maskable.numel()), "total_papers": int(P),
        "diagnostics_explained": {
            "node_eff_rank": "eff_rank of raw GIN node embeddings (~18 = healthy ceiling).",
            "patchmean_eff_rank": "eff_rank of pooled patch vectors; MEAN pooling collapses this to ~2.",
            "pool_eff_rank": "eff_rank of the retrieval target pool.",
            "sanity": "valid retrieval must keep A1~0.85; A1~0.57 or MRR=1.000 => degenerate artifact.",
        },
        "changelog": {
            "MOD1-5": "VICReg-on-encoder, weighted invariance, dose-response, per-dim hyperbola, "
                      "Lorentz-hyperboloid JEPA + geodesic retrieval, 3-level rank probe.",
            "MOD6a_attn_pool": "learned attention pooling (convex combination -> still concentrates).",
            "MOD6b_multistat_pool": "concat [mean||max||std] -> Linear -> LayerNorm.",
            "MOD6c_pooling_sweep": "every objective x every pooling mode + 3-level rank probe.",
            "PATCH_v5.1": "robust eff_rank (float64 + ridge + SVD fallback) fixes eigvalsh err 129.",
            "PATCH_v5.2": "MultiStatPool + LayerNorm + std clamp fixes NaN rank & spurious MRR=1.000; "
                          "NaN-safe rank aggregation.",
        },
    }, open(RESULTS_JSON, "w"), indent=2)
    print(f"[save] {RESULTS_JSON}")


if __name__ == "__main__":
    main()