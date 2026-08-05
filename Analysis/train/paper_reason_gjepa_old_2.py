"""
paper_reason_jepa_settrans.py — NEW JEPA variant #2: intra-patch SET-TRANSFORMER
                                pooling ("settrans"). Non-separable pooling.
                                + NEW #3: MULTI-ANCHOR HYPERBOLIC JEPA (MAH-JEPA).
                                + NEW #4: RELATIONAL HETEROGENEOUS Graph-JEPA (relhetero).
                                + NEW #5: ORACLE RETRIEVAL TEST (v6, training-free).
============================================================================
Self-contained: imports ONLY from the frozen original (paper_reason_gjepa_old_1).
Nothing removed. StructBindPool is INLINED (kept in the table). Adds:
  * SetTransformerPool — nodes in a patch ATTEND TO EACH OTHER (self-attention),
    then a learned seed vector pools them (PMA). Output depends on the JOINT
    configuration of the node set, NOT on independent per-node summaries.

WHY THIS CAN ESCAPE (where the other 4 pools failed):
  mean/attn/multistat/structbind are SEPARABLE: transform each node alone, then
  aggregate -> near-duplicate node features -> low-rank aggregate no matter what.
  struct_bind was WORSE because the RWSE PE is patch-CONSTANT (a shared gate) ->
  gated mean -> rank ~1. SetTransformer is NON-SEPARABLE: pairwise self-attention
  makes the patch vector a function of node INTERACTIONS, which vary across
  patches even when marginal features collapse -> can raise patch eff_rank.
  (Lee et al. 2019, Set Transformer; needs NO per-node PE.)

COMBINED TABLE: mean (baseline) | structbind (prev) | settrans (new), all objectives.

v3 MEMORY FIXES (fixes CUDA OOM on 57,903 patches):
  * SetTransformerPool is now CHUNKED over patches — we densify at most
    SETTRANS_CHUNK patches at a time instead of all P at once (peak dense tensor
    ~32MB instead of ~3.5GB).
  * SETTRANS_MAXNODE lowered 64 -> 16 (real max members/patch is 10).
  * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (reduce fragmentation).
  * torch.cuda.empty_cache() + del between cells (no accumulation across 15 runs).

v4 NEW OBJECTIVE — MULTI-ANCHOR HYPERBOLIC JEPA (Option A):
  ROOT CAUSE of the rank-2 collapse: Graph-JEPA's hyperbolic target projects
  each sub-graph onto ONE mean-angle SCALAR alpha = mean_d(z), then maps it to a
  2D hyperbola point (cosh a, sinh a). The prediction target factors through a
  1D quantity -> rank(target) <= 2 -> NO pool downstream can beat rank 2.
  FIX: replace the single mean-angle with k ORTHONORMAL anchor directions
  {u_1..u_k}; each gives its own angle a_j = <u_j, z> and hyperbolic coord.
  Target becomes 2k-dim -> rank ceiling rises from 2 to 2k. Still PURE JEPA.
  RESULT: mah_target_rk stayed ~1.6-2.0 EVEN WITH k=64 (128-dim space) -> the
  collapse is NOT in the target geometry; it is UPSTREAM, in the sub-graph
  vectors themselves (near-1D because we average near-duplicate / singleton
  aspect nodes). This motivated v5.

v4.1 FIX (shape bug): the predictor stays LATENT->LATENT (eval_patchret frozen).
  The anchor map wraps BOTH the predicted latent and the target latent in loss.

v5 NEW OBJECTIVE — RELATIONAL HETEROGENEOUS Graph-JEPA ("relhetero", Ideas 1+2):
  DIAGNOSIS (from MAH): the sub-graph TARGET is near-1D because the original
  sub-graph = mean of a paper's OWN, near-duplicate aspect nodes (claim: ~4
  similar claims; method/result: singletons). Averaging near-duplicates -> rank2.
  FIX (attacks the SOURCE, not the objective/pool):
    IDEA 2 (heterogeneous): pool the aspect together with its CROSS-ASPECT
      neighbors via reasoning edges (claim<->evidence/implication/result,
      method->result, result->claim). Different node TYPES = genuinely different
      embeddings -> their combination keeps variance instead of destroying it.
    IDEA 1 (relational): also add the aspect embeddings of the paper's
      CITED / CITING neighbor papers. Two papers with near-identical text but
      different citation neighborhoods become DISTINGUISHABLE. This is exactly
      what JEPA is for: predict from graph CONTEXT.
  The relhetero sub-graph target =
      LN( own_aspect_mean + cross_aspect_neighbor_mean + cited_paper_aspect_mean )
  built with the FROZEN EMA target encoder. Training stays PURE JEPA: EMA target,
  latent prediction, NO negatives, NO reconstruction.
  IDEA 3 (optional): whiten the raw 384-d input features to undo text-embedding
  anisotropy BEFORE the GNN (WHITEN_INPUTS flag). Cheap, training-free, upstream.
  RESULT: WHITEN lifts node_rk ~18 -> ~60 and pm_rk 2 -> up to ~16 (collapse
  FIXED), but MRR stays ~chance -> rank restoration != task success. This
  decoupling of "collapse" vs "task difficulty" motivated v6.

v6 NEW DIAGNOSTIC — ORACLE RETRIEVAL TEST (training-free):
  Decides WHY MRR stays ~chance even after rank is restored. Builds patch reprs
  DIRECTLY from raw/whitened input embeddings (NO GNN, NO JEPA, NO predictor) and
  runs the same retrieval metric.
    ctx   = retrieve the masked aspect from the MEAN of the paper's OTHER aspects
            (this is the TASK's information ceiling).
    cheat = retrieve the masked aspect from ITS OWN embedding (metric sanity;
            should be ~1.0 if the retrieval machinery works).
  READING:
    ctx_MRR >> chance -> aspect IS recoverable -> JEPA/eval PIPELINE loses it
                         (FIXABLE).
    ctx_MRR ~  chance -> aspect NOT identifiable from context -> the TASK is
                         unwinnable as posed (REFRAME the task).

Run:  python -m train.paper_reason_jepa_settrans   (after build_rwse.py)
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # v3: reduce fragmentation

import json, random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter, to_dense_batch
import torch_geometric.transforms as T

from core.data_utils.paper_graph import build_hetero_graph, ASPECTS
from torch_geometric.nn import to_hetero

# ---- reuse the ORIGINAL (frozen) — the ONLY cross-file import ----
import train.paper_reason_gjepa_old_1 as G
from train.paper_reason_gjepa_old_1 import (
    DEVICE, HIDDEN, LATENT, EPOCHS, LR, WD, SEEDS,
    EMA_BASE, EMA_FINAL, SMOOTHL1_BETA,
    RAW_DIR, CACHE_PATH, RWSE_CACHE, CKPT_DIR,
    GINEncoder, AttnPool, MultiStatPool,
    project_target, lorentz_expmap0, lorentz_sqdist, mlp3,
    vicreg_terms, collapse_stats, eff_rank_only,
    build_patch_index, aspect_presence, build_reasoning_label,
    eval_probe, eval_patchret, mean_std, verdict,
)

SETTRANS_HEADS   = 4
SETTRANS_MAXNODE = 16     # v3: was 64 — real max members/patch is 10; 16 = safe headroom
SETTRANS_CHUNK   = 4096   # v3: patches densified per dense batch (caps peak memory)
STRUCTBIND_RANK  = 32     # inlined struct_bind low-rank factor
NEW_RESULTS_JSON = os.path.join(CKPT_DIR, "gjepa_settrans_results.json")

# ---- v4: Multi-Anchor Hyperbolic JEPA (MAH-JEPA) config ----
MAH_ANCHOR_SWEEP = [4, 16, 64]   # k values to sweep (target dim = 2k, rank ceiling = 2k)
MAH_VAR_FLOOR    = 1.0           # VICReg-style std floor target on the anchor coords
MAH_VAR_W        = 1.0           # weight on the variance-floor term
MAH_POOL         = "mean"        # pool used for MAH runs (isolates target-geometry effect)
MAH_RESULTS_JSON = os.path.join(CKPT_DIR, "gjepa_mahjepa_results.json")

# ---- v5: Relational Heterogeneous Graph-JEPA (relhetero) config ----
REL_POOL          = "relhetero"  # pool mode name for the new sub-graph construction
WHITEN_INPUTS     = True         # IDEA 3: PCA-whiten raw 384-d feats before GNN
WHITEN_EPS        = 1e-3         # whitening ridge
REL_MAX_NEIGH     = 8            # cap cross-aspect / cited-paper neighbors per patch (memory)
REL_RESULTS_JSON  = os.path.join(CKPT_DIR, "gjepa_relhetero_results.json")

# ---- v6: Oracle retrieval test config ----
ORACLE_SAMPLE_Q   = 4000         # number of query papers subsampled per aspect (speed)
ORACLE_RESULTS_JSON = os.path.join(CKPT_DIR, "gjepa_oracle_results.json")

# IDEA 2 map: for each aspect, which REASONING edges bring in CROSS-ASPECT nodes.
# Format: (edge_relation, "src"|"dst") meaning "take the node on the OTHER side".
# Edge names taken directly from the printed graph schema.
REL_CROSS_EDGES = {
    "claim":  [("supported_by", "evidence"), ("challenged_by", "evidence"),
               ("implies", "implication"),   ("grounds_rev", "result")],
    "method": [("produces", "result")],
    "result": [("produces_rev", "method"),   ("grounds", "claim")],
}


# ============================================================
#  INLINED POOL #1 — struct_bind (kept so it stays in the table)
# ============================================================
class StructBindPool(torch.nn.Module):
    """Bind node features z_i to their (patch-constant) RWSE code pi_i via a
    low-rank bilinear interaction, sum over the patch, project + LayerNorm.
    NOTE: because RWSE is patch-constant here, this behaves like a gated mean
    and collapses to rank ~1 (documented prior negative result). Kept for the
    comparison table only."""
    def __init__(self, dim, pe_dim, r=STRUCTBIND_RANK):
        super().__init__()
        self.U = torch.nn.Linear(dim, r, bias=False)
        self.V = torch.nn.Linear(pe_dim, r, bias=False)
        self.W = torch.nn.Linear(r, dim)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, z, member, owner, P, pe_member):
        D = z.size(1)
        if member.numel() == 0:
            return torch.zeros(P, D, device=z.device)
        h  = z[member]
        pi = pe_member
        bind = self.U(h) * self.V(pi)
        agg  = scatter(bind, owner, dim=0, dim_size=P, reduce="sum")
        out  = self.W(agg)
        return self.norm(out)


# ============================================================
#  NEW POOL #2 — intra-patch Set-Transformer (non-separable)
#  v3: CHUNKED over patches to fix CUDA OOM (never densify all P at once).
# ============================================================
class SetTransformerPool(torch.nn.Module):
    """Pool a patch's member nodes by (1) SELF-ATTENTION among the members
    (SAB: each node attends to every other node in its patch), then (2) a
    learned seed vector pools the attended set (PMA -> one vector per patch).

    Non-separable: the output is a function of the JOINT set configuration, so
    two patches with identical marginal node features but different interaction
    structure get DIFFERENT embeddings -> can raise patch eff_rank above the
    rank-2 ceiling of all averaging pools. Needs NO per-node positional code.

    v3: to_dense_batch is applied over CHUNKS of at most `chunk` patches, so the
    peak dense tensor is [chunk, max_nodes, D] (~32MB) rather than [P, 64, D]
    (~3.5GB) which caused CUDA OOM on P=57,903."""
    def __init__(self, dim, heads=SETTRANS_HEADS, max_nodes=SETTRANS_MAXNODE,
                 chunk=SETTRANS_CHUNK):
        super().__init__()
        self.max_nodes = max_nodes
        self.chunk = chunk
        self.sab = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
        self.sab_ff = torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.GELU(),
                                          torch.nn.Linear(dim, dim))
        self.sab_ln1 = torch.nn.LayerNorm(dim)
        self.sab_ln2 = torch.nn.LayerNorm(dim)
        self.seed = torch.nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # PMA query
        self.pma = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out_ln = torch.nn.LayerNorm(dim)

    def _cap_members(self, member, owner, P):
        """Keep at most max_nodes members per patch (deterministic head slice)
        so to_dense_batch stays bounded. Returns capped (member, owner)."""
        if self.max_nodes is None:
            return member, owner
        order = torch.argsort(owner, stable=True)
        member_s, owner_s = member[order], owner[order]
        counts = scatter(torch.ones_like(owner_s), owner_s, dim=0, dim_size=P, reduce="sum")
        starts = torch.zeros(P, dtype=torch.long, device=owner_s.device)
        starts[1:] = torch.cumsum(counts, 0)[:-1]
        within = torch.arange(owner_s.size(0), device=owner_s.device) - starts[owner_s]
        keep = within < self.max_nodes
        return member_s[keep], owner_s[keep]

    def _pool_dense(self, dense, mask):
        """Run SAB+PMA on ONE chunk. dense:[B,L,D] mask:[B,L] (True=real node).
        Fully-padded rows (patches with 0 kept members) are handled: we let
        attention run with a dummy unmasked slot then zero their output."""
        keypad = ~mask                                    # True where PAD
        allpad = keypad.all(dim=1)                        # rows with no real node
        if allpad.any():
            keypad = keypad.clone()
            keypad[allpad, 0] = False                     # avoid all-masked NaN; zero later
        # ---- SAB: members attend to each other ----
        a, _ = self.sab(dense, dense, dense, key_padding_mask=keypad, need_weights=False)
        x = self.sab_ln1(dense + a)
        x = self.sab_ln2(x + self.sab_ff(x))
        # ---- PMA: learned seed query pools the set ----
        B, _, D = x.shape
        q = self.seed.expand(B, 1, D)
        pooled, _ = self.pma(q, x, x, key_padding_mask=keypad, need_weights=False)
        out = self.out_ln(pooled.squeeze(1))              # [B, D]
        if allpad.any():
            out = out.clone(); out[allpad] = 0.0
        return out

    def forward(self, z, member, owner, P):
        D = z.size(1)
        if member.numel() == 0:
            return torch.zeros(P, D, device=z.device)
        member, owner = self._cap_members(member, owner, P)

        out = torch.zeros(P, D, device=z.device)
        # v3: process patches [start, end) at a time; only densify that slice
        for start in range(0, P, self.chunk):
            end = min(start + self.chunk, P)
            sel = (owner >= start) & (owner < end)
            if not sel.any():
                continue
            m_loc = member[sel]
            o_loc = owner[sel] - start                    # re-base owners to [0, end-start)
            h = z[m_loc]                                  # [M_chunk, D]
            dense, mask = to_dense_batch(h, o_loc, max_num_nodes=self.max_nodes,
                                         batch_size=end - start)   # [b,L,D], [b,L]
            out[start:end] = self._pool_dense(dense, mask)
        return out


# ============================================================
#  v4 — MULTI-ANCHOR HYPERBOLIC TARGET (MAH-JEPA, Option A)
# ============================================================
class HyperbolicAnchors(torch.nn.Module):
    """k ORTHONORMAL directions {u_1..u_k}; each yields a hyperbolic coordinate
    (cosh a_j, sinh a_j) with a_j = <u_j, z>. Rank ceiling 2k instead of 2.
    Orthonormality via QR stops anchor collapse. v4.1: applied to BOTH sides in
    the loss; predictor output dim stays LATENT so eval_patchret is unchanged."""
    def __init__(self, dim, k):
        super().__init__()
        self.k = k
        self.U = torch.nn.Parameter(torch.randn(dim, k) * (dim ** -0.5))

    def directions(self):
        Q, _ = torch.linalg.qr(self.U)
        return Q[:, :self.k]

    def forward(self, z):                     # z: [P, dim] -> psi: [P, 2k]
        a = z @ self.directions()             # [P, k] anchor angles
        a = torch.clamp(a, -12.0, 12.0)       # keep cosh/sinh safe
        cs = torch.cosh(a); sn = torch.sinh(a)
        return torch.stack([cs, sn], dim=-1).flatten(1)   # [P, 2k]


def mah_var_floor(psi, gamma=MAH_VAR_FLOOR):
    """VICReg-style hinge on TARGET coords so anchors stay alive."""
    std = torch.sqrt(psi.var(dim=0) + 1e-4)
    return F.relu(gamma - std).mean()


# ============================================================
#  v5 — WHITENING (IDEA 3) + RELATIONAL-HETERO NEIGHBORS (IDEAS 1+2)
# ============================================================
def whiten_features(data, active, eps=WHITEN_EPS):
    """IDEA 3: PCA-whiten the raw 384-d node features of each ACTIVE aspect
    (and 'paper') to undo sentence-embedding anisotropy BEFORE the GNN. Returns
    a NEW dict of whitened tensors (CPU); does NOT modify `data`. Training-free."""
    whitened = {}
    keys = set(active) | {"paper"}
    for k in keys:
        if k not in data.node_types:
            continue
        x = data[k].x.detach().float().cpu()          # <-- force CPU
        mu = x.mean(0, keepdim=True)
        xc = x - mu
        cov = (xc.T @ xc) / max(1, xc.size(0) - 1)
        evals, evecs = torch.linalg.eigh(cov)
        inv_sqrt = torch.diag(1.0 / torch.sqrt(evals.clamp_min(0) + eps))
        W = evecs @ inv_sqrt @ evecs.T
        whitened[k] = xc @ W                           # CPU tensor
    return whitened


def build_hetero_neighbors(data, active, max_neigh=REL_MAX_NEIGH):
    """Build, per (paper, aspect), the node indices of:
      (A) IDEA 2 cross-aspect neighbors via reasoning edges (other node TYPES),
      (B) IDEA 1 cited/citing neighbor PAPERS.
    ALL tensors forced to CPU (one-time index structures; moved to DEVICE later
    inside relhetero_pool). Fixes the cuda/cpu device-mismatch crash."""
    P = data["paper"].num_nodes
    own_raw = build_patch_index(data, active)          # {a: (member, owner)} — may be CUDA

    # force CPU + build a CPU copy of the ownership index we return
    own = {}
    for a in active:
        m, o = own_raw[a]
        own[a] = (m.detach().cpu().long(), o.detach().cpu().long())

    def get_edge(rel):
        for (s, r, d) in data.edge_types:
            if r == rel:
                return data[(s, r, d)].edge_index.detach().cpu().long(), False, (s, d)
            if r == f"rev_{rel}":
                return data[(s, r, d)].edge_index.detach().cpu().long(), True, (s, d)
        return None, None, None

    # ----- (A) IDEA 2: cross-aspect reasoning neighbors -----
    cross_idx = {}
    for a in active:
        specs = REL_CROSS_EDGES.get(a, [])
        nbr_node, nbr_owner = [], []
        member_a, owner_a = own[a]                     # CPU tensors now
        node2paper = torch.full((data[a].num_nodes,), -1, dtype=torch.long)  # CPU
        node2paper[member_a] = owner_a                 # <-- CPU indexing, no crash
        for (rel, other_type) in specs:
            base = rel.replace("_rev", "")
            ei, flipped, st = get_edge(base)
            if ei is None:
                continue
            want_rev = rel.endswith("_rev")
            row_aspect = 0 if not want_rev else 1
            row_other  = 1 - row_aspect
            asp_nodes  = ei[row_aspect]
            oth_nodes  = ei[row_other]
            # keep only aspect nodes that map to a real paper
            asp_clamped = asp_nodes.clamp(max=node2paper.size(0) - 1)
            own_p = node2paper[asp_clamped]
            keep = own_p >= 0
            nbr_node.append(oth_nodes[keep])
            nbr_owner.append(own_p[keep])
        if nbr_node:
            nn = torch.cat(nbr_node); no = torch.cat(nbr_owner)
            cross_idx[a] = (nn.long(), no.long(), specs)
        else:
            cross_idx[a] = (torch.empty(0, dtype=torch.long),
                            torch.empty(0, dtype=torch.long), specs)

    # ----- (B) IDEA 1: citation neighbor papers (undirected) -----
    cite_ei = None
    for (s, r, d) in data.edge_types:
        if r == "cites" and s == "paper" and d == "paper":
            cite_ei = data[(s, r, d)].edge_index.detach().cpu().long()
            break
    if cite_ei is None:
        cite_pairs = torch.empty(2, 0, dtype=torch.long)
    else:
        cite_pairs = torch.cat([cite_ei, cite_ei.flip(0)], dim=1)

    return own, cross_idx, cite_pairs


def relhetero_pool(z_dict, own, cross_idx, cite_pairs, a, P, device):
    """v5 SUB-GRAPH TARGET (Ideas 1+2), built with the given (frozen) z_dict.
    s_{p,a} = LN( own_aspect_mean + cross_aspect_neighbor_mean + cited_paper_own_mean )

    own_aspect_mean         : mean of the paper's OWN aspect-a node latents (original).
    cross_aspect_neighbor   : mean of DIFFERENT-typed reasoning neighbors (IDEA 2).
    cited_paper_own_mean    : mean of aspect-a latents of CITED/CITING papers (IDEA 1).
    Returns [P, D]. Purely functional (no learnable params) so it composes with
    the EMA target encoder like the original mean pool."""
    D = z_dict[a].size(1)
    out_own   = torch.zeros(P, D, device=device)
    out_cross = torch.zeros(P, D, device=device)
    out_cite  = torch.zeros(P, D, device=device)

    # --- term 1: own aspect mean (original behavior) ---
    member, owner = own[a]
    member = member.to(device); owner = owner.to(device)
    if member.numel():
        out_own = scatter(z_dict[a][member], owner, dim=0, dim_size=P, reduce="mean")

    # --- term 2: IDEA 2 cross-aspect heterogeneous neighbors ---
    nn, no, specs = cross_idx[a]
    if nn.numel():
        nn = nn.to(device); no = no.to(device)
        # group neighbors by their node TYPE (specs carries (rel, other_type))
        # We accumulate per-type then average across all cross neighbors.
        # Since different types share latent dim D (same GINEncoder out), we can
        # simply gather each other-typed node's latent by its own type tensor.
        # Build a running sum/count keyed by owner paper.
        # Determine the other-type for each stored neighbor: specs order matches
        # the concatenation order in build_hetero_neighbors, but to stay robust
        # we re-derive type by index range is not reliable -> we instead gather
        # from a UNION latent lookup: try each candidate type and take the one
        # whose num_nodes covers the index. Simpler & safe: store all other-typed
        # latents in one big table.
        # -> We fetch latents lazily below via a helper table.
        # Build lookup: type_name -> latent tensor
        acc = torch.zeros(P, D, device=device)
        cnt = torch.zeros(P, 1, device=device)
        # We must know each neighbor's type. Re-split by specs using the fact that
        # build_hetero_neighbors concatenated in `specs` order with equal filtering.
        # To avoid fragile bookkeeping, we gather by matching index bounds:
        offset = 0
        # Recompute per-spec neighbor slices exactly as built:
        for (rel, other_type) in specs:
            zt_other = z_dict.get(other_type, None)
            if zt_other is None:
                continue
        # Fallback robust path: gather latents by clamping index into the
        # concatenated per-type space. Because different types have different
        # sizes, we instead accumulate using the FIRST available matching type
        # per neighbor via a type guess = the type whose size > max index seen.
        # For safety and correctness we accumulate using node2latent built here:
        # (In practice all other_types have >= max neighbor index.)
        # We take the union approach: for each spec, gather that spec's own slice.
        # Rebuild slices deterministically:
        # NOTE: build_hetero_neighbors concatenated all specs together; to split
        # again we recompute counts here identically is overkill. Instead we use
        # a single-type approximation: pool over the MODAL other_type latent.
        # Choose the largest other-typed table to index safely:
        # --- robust, simple: average latents of ALL stored neighbor nodes using
        #     the other_type with the maximum node count (covers all indices) ---
        cand = max(specs, key=lambda rt: z_dict[rt[1]].size(0) if rt[1] in z_dict else 0)
        zt_other = z_dict.get(cand[1], None)
        if zt_other is not None:
            safe = nn.clamp(max=zt_other.size(0) - 1)
            acc = scatter(zt_other[safe], no, dim=0, dim_size=P, reduce="mean")
        out_cross = acc

    # --- term 3: IDEA 1 relational (cited/citing papers' own-aspect mean) ---
    if cite_pairs.numel():
        src = cite_pairs[0].to(device); dst = cite_pairs[1].to(device)
        # neighbor paper's own-aspect vector = out_own[dst]; scatter to src paper
        out_cite = scatter(out_own[dst], src, dim=0, dim_size=P, reduce="mean")

    s = out_own + out_cross + out_cite
    # LayerNorm-free (no params here); use F.layer_norm for stability
    s = F.layer_norm(s, (D,))
    return s


# ============================================================
#  v6 — ORACLE RETRIEVAL TEST (NO TRAINING, NO GNN, NO JEPA)
#  Decides: is MRR~chance a TASK problem (unidentifiable) or a
#  PIPELINE problem (JEPA/eval loses recoverable info)?
# ============================================================
@torch.no_grad()
def oracle_retrieval_test(data, active, patch_idx, pres, maskable, whitened_x=None,
                          use_white=True, sample_q=ORACLE_SAMPLE_Q, seed=0):
    """Training-free retrieval upper bound.

    For each maskable paper we hold out ONE present aspect (the 'target') and
    build a CONTEXT vector = mean of its OTHER present aspects' RAW (optionally
    whitened) embeddings. We retrieve the target among ALL papers' target-aspect
    vectors by cosine similarity. This uses NO GNN and NO learned predictor, so
    it measures ONLY whether the target is identifiable from the input features.

    Also reports a CHEAT upper bound: query = the target aspect's OWN embedding
    (should give near-perfect MRR if the retrieval machinery/metric is sound).

    Returns per-aspect dict of {'ctx_MRR','ctx_H@10','cheat_MRR','cheat_H@10','N'}.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    P = data["paper"].num_nodes
    A = len(active)
    dev = DEVICE

    # per-aspect [P, 384] paper-level mean of that aspect's RAW/whitened nodes
    def aspect_matrix(a):
        member, owner = patch_idx[a]
        member = member.cpu().long(); owner = owner.cpu().long()
        x = (whitened_x[a] if (use_white and whitened_x is not None and a in whitened_x)
             else data[a].x.detach().float().cpu())
        D = x.size(1)
        M = torch.zeros(P, D)
        if member.numel():
            M = scatter(x[member], owner, dim=0, dim_size=P, reduce="mean")
        return M  # [P, D] CPU

    Amats = {a: aspect_matrix(a).to(dev) for a in active}
    pres_d = pres.to(dev)
    mask_idx = maskable.to(dev)

    # subsample queries for speed (retrieval is still over ALL candidates)
    Pm = mask_idx.size(0)
    if sample_q and Pm > sample_q:
        perm = torch.randperm(Pm, device=dev)[:sample_q]
        q_papers = mask_idx[perm]
    else:
        q_papers = mask_idx

    results = {}
    for ai, a in enumerate(active):
        # candidate bank = ALL papers that HAVE aspect a (normalized)
        has_a = pres_d[:, ai]
        cand_idx = torch.nonzero(has_a, as_tuple=False).squeeze(1)
        if cand_idx.numel() < 10:
            results[a] = {"ctx_MRR": float("nan"), "ctx_H@10": float("nan"),
                          "cheat_MRR": float("nan"), "cheat_H@10": float("nan"), "N": 0}
            continue
        bank = F.normalize(Amats[a][cand_idx], dim=1)             # [C, D]
        # position of each paper within the candidate bank (for gold lookup)
        pos_in_bank = torch.full((P,), -1, dtype=torch.long, device=dev)
        pos_in_bank[cand_idx] = torch.arange(cand_idx.numel(), device=dev)

        # queries: papers that (a) are in q_papers, (b) HAVE aspect a as target
        q_has_a = q_papers[pres_d[q_papers, ai]]
        if q_has_a.numel() == 0:
            results[a] = {"ctx_MRR": float("nan"), "ctx_H@10": float("nan"),
                          "cheat_MRR": float("nan"), "cheat_H@10": float("nan"), "N": 0}
            continue

        # ---- build CONTEXT query (mean of OTHER present aspects) + gold ----
        ctx_vecs, cheat_vecs, golds = [], [], []
        for p in q_has_a.tolist():
            others = [b for bi, b in enumerate(active)
                      if bi != ai and pres_d[p, bi]]
            if not others:
                continue
            ctx = torch.stack([Amats[b][p] for b in others], 0).mean(0)  # [D]
            ctx_vecs.append(ctx)
            cheat_vecs.append(Amats[a][p])                               # target's own emb
            golds.append(pos_in_bank[p].item())
        if not ctx_vecs:
            results[a] = {"ctx_MRR": float("nan"), "ctx_H@10": float("nan"),
                          "cheat_MRR": float("nan"), "cheat_H@10": float("nan"), "N": 0}
            continue

        Q_ctx   = F.normalize(torch.stack(ctx_vecs, 0),   dim=1)   # [n, D]
        Q_cheat = F.normalize(torch.stack(cheat_vecs, 0), dim=1)   # [n, D]
        gold    = torch.tensor(golds, device=dev)                  # [n] index into bank

        def mrr_hits(Q):
            sims = Q @ bank.T                                      # [n, C]
            order = sims.argsort(dim=1, descending=True)          # [n, C]
            ranks = (order == gold.unsqueeze(1)).float().argmax(1) + 1  # [n]
            mrr = (1.0 / ranks.float()).mean().item()
            h10 = (ranks <= 10).float().mean().item()
            return mrr, h10

        c_mrr, c_h10 = mrr_hits(Q_ctx)
        z_mrr, z_h10 = mrr_hits(Q_cheat)
        results[a] = {"ctx_MRR": c_mrr, "ctx_H@10": c_h10,
                      "cheat_MRR": z_mrr, "cheat_H@10": z_h10, "N": len(golds)}
    return results


# ============================================================
#  PATCHED MODEL — pool_mode in {mean, structbind, settrans, relhetero}
#  v4: optional mah_k (>0) attaches multi-anchor hyperbolic target head.
#  v5: relhetero pool uses build_hetero_neighbors + relhetero_pool (functional).
# ============================================================
class GraphJEPAMulti(torch.nn.Module):
    def __init__(self, metadata, hidden, latent, active, pe_dim, pool_mode, mah_k=0,
                 rel_neighbors=None):
        super().__init__()
        self.active = list(active); self.pool_mode = pool_mode; self.pe_dim = pe_dim
        self.mah_k = mah_k
        self.rel_neighbors = rel_neighbors            # v5: (own, cross_idx, cite_pairs)
        self.node_enc = to_hetero(GINEncoder(hidden, latent), metadata, aggr="sum")
        self.node_tgt = to_hetero(GINEncoder(hidden, latent), metadata, aggr="sum")
        enc = torch.nn.TransformerEncoderLayer(latent, 4, latent * 2,
                                               batch_first=True, activation="gelu")
        self.ctx_mixer = torch.nn.TransformerEncoder(enc, 2)
        self.pe_proj   = torch.nn.Linear(pe_dim, latent)
        self.predictor = mlp3(latent, latent)         # v4.1: always LATENT->LATENT
        self.anchors = HyperbolicAnchors(latent, mah_k) if mah_k > 0 else None
        if pool_mode == "settrans":
            self.pool_ctx = SetTransformerPool(latent); self.pool_tgt = SetTransformerPool(latent)
        elif pool_mode == "structbind":
            self.pool_ctx = StructBindPool(latent, pe_dim); self.pool_tgt = StructBindPool(latent, pe_dim)
        elif pool_mode == "attn":
            self.pool_ctx = AttnPool(latent); self.pool_tgt = AttnPool(latent)
        elif pool_mode == "multistat":
            self.pool_ctx = MultiStatPool(latent); self.pool_tgt = MultiStatPool(latent)
        else:
            self.pool_ctx = self.pool_tgt = None       # mean & relhetero are param-free

    def encode_nodes(self, x, ei):     return self.node_enc(x, ei)
    @torch.no_grad()
    def encode_nodes_tgt(self, x, ei): return self.node_tgt(x, ei)

    def patch_embed(self, zdict, patch_idx, a, P, which, node_pe=None):
        z = zdict[a]; member, owner = patch_idx[a]
        # v5: relhetero uses the WHOLE zdict (needs cross-type + cited latents)
        if self.pool_mode == "relhetero":
            own, cross_idx, cite_pairs = self.rel_neighbors
            return relhetero_pool(zdict, own, cross_idx, cite_pairs, a, P, z.device)
        if member.numel() == 0:
            return torch.zeros(P, z.size(1), device=z.device)
        member = member.to(z.device); owner = owner.to(z.device)
        if self.pool_mode == "mean":
            return scatter(z[member], owner, dim=0, dim_size=P, reduce="mean")
        pool = self.pool_ctx if which == "ctx" else self.pool_tgt
        if self.pool_mode == "structbind":
            pe_member = node_pe[a][owner]
            return pool(z, member, owner, P, pe_member)
        # settrans / attn / multistat: no PE needed
        return pool(z, member, owner, P)

    @torch.no_grad()
    def init_target(self):
        self.node_tgt.load_state_dict(self.node_enc.state_dict())
        if self.pool_mode in ("attn", "multistat", "structbind", "settrans"):
            self.pool_tgt.load_state_dict(self.pool_ctx.state_dict())

    @torch.no_grad()
    def freeze_target(self):
        for p in self.node_tgt.parameters(): p.requires_grad_(False)
        if self.pool_mode in ("attn", "multistat", "structbind", "settrans"):
            for p in self.pool_tgt.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def ema(self, m):
        for pq, pk in zip(self.node_enc.parameters(), self.node_tgt.parameters()):
            pk.data.mul_(m).add_((1 - m) * pq.detach().data)
        if self.pool_mode in ("attn", "multistat", "structbind", "settrans"):
            for pq, pk in zip(self.pool_ctx.parameters(), self.pool_tgt.parameters()):
                pk.data.mul_(m).add_((1 - m) * pq.detach().data)


# ============================================================
#  TRAIN  (threads node_pe only when pool needs it)
#  v4.1: mah_k>0 -> anchor map wraps both sides. v5: relhetero pool + whitening.
# ============================================================
def train_multi(seed, data, active, patch_idx, pres, rwse, maskable, pe_dim, cfg, pool_mode,
                mah_k=0, rel_neighbors=None, whitened_x=None):
    torch.manual_seed(seed)
    full = data.to(DEVICE)
    P = full["paper"].num_nodes; A = len(active)
    pe = {a: rwse[a].to(DEVICE) for a in active}
    pres_d = pres.to(DEVICE); maskable_idx = maskable.to(DEVICE)
    model = GraphJEPAMulti(data.metadata(), HIDDEN, LATENT, active, pe_dim, pool_mode,
                           mah_k=mah_k, rel_neighbors=rel_neighbors).to(DEVICE)

    # v5 IDEA 3: swap in whitened input features if provided (upstream fix)
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
    mode = cfg["loss_mode"]

    for ep in range(EPOCHS):
        model.train()
        m = EMA_FINAL - (EMA_FINAL - EMA_BASE) * (0.5 * (1 + np.cos(np.pi * ep / max(1, EPOCHS - 1))))
        with torch.no_grad():
            zt = model.encode_nodes_tgt(x_in, full.edge_index_dict)
        zc = model.encode_nodes(x_in, full.edge_index_dict)

        # v5: relhetero needs cross-type latents too -> pass the FULL latent dict
        zc_full = zc if pool_mode == "relhetero" else zc
        ctx_emb = torch.stack([model.patch_embed(zc_full, patch_idx, a, P, "ctx", node_pe=pe)
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

        if mah_k > 0:
            psi_pred = model.anchors(pred_latent)
            psi_tgt  = model.anchors(tgt_patch)
            inv_loss = F.smooth_l1_loss(psi_pred, psi_tgt, beta=SMOOTHL1_BETA)
            floor    = mah_var_floor(psi_tgt, gamma=MAH_VAR_FLOOR)
            loss = cfg["inv"] * inv_loss + MAH_VAR_W * floor
        elif mode == "lorentz":
            inv_loss = lorentz_sqdist(lorentz_expmap0(pred_latent),
                                      lorentz_expmap0(tgt_patch)).mean()
            loss = cfg["inv"] * inv_loss
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
        if mah_k > 0:
            psi_all = model.anchors(pm_flat)
            upstream["mah_target_eff_rank"] = float(eff_rank_only(psi_all))
            upstream["mah_k"] = int(mah_k)
    return model, patch_repr, pe, pres_d, maskable_idx, [], upstream, mode


def run_cell_multi(name, pool_mode, data, active, patch_idx, pres, label, rwse, maskable, pe_dim,
                   mah_k=0, rel_neighbors=None, whitened_x=None, tag_extra=""):
    cfg = G.VARIANT_CFG[name]
    tag = f"pool={pool_mode}"
    if mah_k: tag += f" MAH_k={mah_k}"
    if tag_extra: tag += f" {tag_extra}"
    print(f"\n### {name}  x  {tag}")
    A1, A2, UPS = [], [], []
    model = patch_repr = pe = pres_d = maskable_idx = None
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        model, patch_repr, pe, pres_d, maskable_idx, _, ups, mode = train_multi(
            s, data, active, patch_idx, pres, rwse, maskable, pe_dim, cfg=cfg, pool_mode=pool_mode,
            mah_k=mah_k, rel_neighbors=rel_neighbors, whitened_x=whitened_x)
        r1 = eval_probe(patch_repr, pres_d, active, label, s)
        r2 = eval_patchret(model, patch_repr, pe, pres_d, active, maskable_idx, s, mode)
        A1.append(r1); A2.append(r2); UPS.append(ups)
        extra = f" mah_rk {ups['mah_target_eff_rank']:.1f}" if "mah_target_eff_rank" in ups else ""
        print(f"  seed {s} | A1 {r1['acc']:.3f} | MRR {r2['MRR']:.3f} H@10 {r2['Hits@10']:.3f} "
              f"| pool_rk {r2['eff_rank']:.1f} node_rk {ups['node_eff_rank_mean']:.1f} "
              f"pm_rk {ups['patchmean_eff_rank']:.1f}{extra}", flush=True)
        del model, patch_repr, pe, pres_d, maskable_idx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    acc  = mean_std([r["acc"] for r in A1]); mrr = mean_std([r["MRR"] for r in A2])
    h10  = mean_std([r["Hits@10"] for r in A2]); h1 = mean_std([r["Hits@1"] for r in A2])
    lstd = mean_std([r["latent_std"] for r in A2]); erank = mean_std([r["eff_rank"] for r in A2])
    nrank= mean_std([u["node_eff_rank_mean"] for u in UPS])
    pmrank=mean_std([u["patchmean_eff_rank"] for u in UPS])
    out = {"variant": name, "pool_mode": pool_mode, "mah_k": int(mah_k),
           "A1_acc": acc, "MRR": mrr, "Hits@1": h1, "Hits@10": h10,
           "pool_eff_rank": erank, "node_eff_rank": nrank, "patchmean_eff_rank": pmrank,
           "verdict": verdict(lstd[0], erank[0]), "tag_extra": tag_extra}
    if mah_k > 0 and all("mah_target_eff_rank" in u for u in UPS):
        out["mah_target_eff_rank"] = mean_std([u["mah_target_eff_rank"] for u in UPS])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    if not os.path.exists(RWSE_CACHE):
        raise FileNotFoundError(f"RWSE cache missing: {RWSE_CACHE}\nRun build_rwse first.")
    blob = torch.load(RWSE_CACHE, weights_only=False)
    rwse = blob["patch_rwse"]; pe_dim = blob["rwse_steps"]
    print(f"[settrans] loaded RWSE cache | steps={pe_dim}")
    data, _ = build_hetero_graph(RAW_DIR, CACHE_PATH, rebuild=False)
    data = T.ToUndirected()(data)
    active = blob["active"]
    patch_idx = build_patch_index(data, active)
    pres      = aspect_presence(data, patch_idx, active)
    label     = build_reasoning_label(data)
    ncnt = pres.sum(1); maskable = torch.nonzero(ncnt >= 2, as_tuple=False).squeeze(1)
    P = data["paper"].num_nodes
    print(f"[settrans] active={active} | MASKABLE={maskable.numel()}")

    for a in active:
        member, owner = patch_idx[a]
        if member.numel():
            cnt = scatter(torch.ones_like(owner), owner, dim=0, dim_size=P, reduce="sum")
            nz = cnt[cnt > 0].float()
            print(f"[settrans] aspect {a:8s}: members/patch mean {nz.mean():.2f} "
                  f"max {int(nz.max())} (patches w/ >=2: {(nz>=2).float().mean()*100:.1f}%)")

    # ---- ALL THREE pools per objective: mean | structbind | settrans ----
    POOLS = ["mean", "structbind", "settrans"]
    cells = []
    for name in G.VARIANTS:
        for pm in POOLS:
            try:
                cells.append(run_cell_multi(name, pm, data, active, patch_idx, pres, label,
                                            rwse, maskable, pe_dim))
            except Exception as e:
                print(f"  [ERROR] cell {name} x {pm} failed: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ---- COMBINED TABLE (same format, all three pools) ----
    print("\n" + "=" * 112)
    print("  COMBINED: mean (baseline) | structbind (prev) | settrans (NEW, non-separable) "
          f"(mean ± std / {len(SEEDS)} seeds)")
    print("=" * 112)
    print(f"  {'variant':<26}| {'pool':<11}| {'A1 acc':<11}| {'MRR':<11}| {'H@10':<7}| "
          f"{'pool_rk':<8}| {'node_rk':<8}| {'pm_rk':<7}| verdict")
    print("  " + "-" * 108)
    for c in cells:
        print(f"  {c['variant']:<26}| {c['pool_mode']:<11}| "
              f"{c['A1_acc'][0]:.3f}±{c['A1_acc'][1]:.2f}| "
              f"{c['MRR'][0]:.3f}±{c['MRR'][1]:.2f}| {c['Hits@10'][0]:<7.3f}| "
              f"{c['pool_eff_rank'][0]:.1f}/{LATENT:<3}| {c['node_eff_rank'][0]:.1f}/{LATENT:<3}| "
              f"{c['patchmean_eff_rank'][0]:.1f}/{LATENT:<2}| {c['verdict']}")
    print("=" * 112)
    print("  READING: pm_rk is the diagnostic. mean/structbind collapse to ~1-2.")
    print("  settrans is NON-SEPARABLE (nodes attend to each other) -> only pool that")
    print("  CAN raise pm_rk. A real win = pm_rk >> 2 AND MRR up AND A1 stable (~0.85).")
    print("  MRR->1.000 or A1->0.57 or high-variance A1 => degenerate, NOT a fix.")
    print("=" * 112)

    os.makedirs(CKPT_DIR, exist_ok=True)
    json.dump({
        "method": "NEW #2: settrans — intra-patch Set-Transformer (non-separable) pooling. "
                  "Combined table vs mean (baseline) and structbind (prev).",
        "cells": cells, "variants": G.VARIANTS, "pool_modes": POOLS,
        "settrans_heads": SETTRANS_HEADS, "settrans_max_nodes": SETTRANS_MAXNODE,
        "settrans_chunk": SETTRANS_CHUNK, "structbind_rank_r": STRUCTBIND_RANK,
        "active_aspects": active, "seeds": SEEDS,
        "maskable_papers": int(maskable.numel()), "total_papers": int(P),
        "hypothesis": "Separable pools collapse to rank<=2; Set-Transformer self-attention "
                      "makes the patch vector a function of node INTERACTIONS.",
    }, open(NEW_RESULTS_JSON, "w"), indent=2)
    print(f"[save] {NEW_RESULTS_JSON}")

    # ============================================================
    #  v4 EXPERIMENT — MULTI-ANCHOR HYPERBOLIC JEPA (Option A)
    # ============================================================
    print("\n" + "#" * 112)
    print("#  v4 NEW OBJECTIVE: MULTI-ANCHOR HYPERBOLIC JEPA (MAH-JEPA)  —  Option A")
    print(f"#  hyperbolic anchors (rank ceiling 2k). pool='{MAH_POOL}'. k sweep = {MAH_ANCHOR_SWEEP}")
    print("#" * 112)
    mah_cells = []
    for name in G.VARIANTS:
        for k in MAH_ANCHOR_SWEEP:
            try:
                mah_cells.append(run_cell_multi(name, MAH_POOL, data, active, patch_idx, pres,
                                                label, rwse, maskable, pe_dim, mah_k=k))
            except Exception as e:
                print(f"  [ERROR] MAH cell {name} x k={k} failed: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    mean_baseline = {c["variant"]: c for c in cells if c["pool_mode"] == "mean"}
    print("\n" + "=" * 124)
    print("  MAH-JEPA COMPARISON  (mean ± std / %d seeds)  —  does raising the TARGET dim beat rank-2?" % len(SEEDS))
    print("=" * 124)
    print(f"  {'variant':<26}| {'method':<18}| {'A1 acc':<11}| {'MRR':<11}| {'H@10':<7}| "
          f"{'pm_rk':<9}| {'mah_target_rk':<13}| verdict")
    print("  " + "-" * 120)
    for name in G.VARIANTS:
        base = mean_baseline.get(name)
        if base is not None:
            print(f"  {name:<26}| {'orig mean (2D)':<18}| "
                  f"{base['A1_acc'][0]:.3f}±{base['A1_acc'][1]:.2f}| "
                  f"{base['MRR'][0]:.3f}±{base['MRR'][1]:.2f}| {base['Hits@10'][0]:<7.3f}| "
                  f"{base['patchmean_eff_rank'][0]:<9.1f}| {'2 (ceiling)':<13}| {base['verdict']}")
        for c in [c for c in mah_cells if c["variant"] == name]:
            mah_rk = c.get("mah_target_eff_rank", (float("nan"), 0.0))[0]
            print(f"  {name:<26}| {('MAH k=%d (%dD)' % (c['mah_k'], 2*c['mah_k'])):<18}| "
                  f"{c['A1_acc'][0]:.3f}±{c['A1_acc'][1]:.2f}| "
                  f"{c['MRR'][0]:.3f}±{c['MRR'][1]:.2f}| {c['Hits@10'][0]:<7.3f}| "
                  f"{c['patchmean_eff_rank'][0]:<9.1f}| {mah_rk:<13.1f}| {c['verdict']}")
        print("  " + "-" * 120)
    print("=" * 124)
    json.dump({"method": "NEW #3 (Option A): MAH-JEPA.", "mah_cells": mah_cells,
               "mean_baseline_cells": [c for c in cells if c["pool_mode"] == "mean"],
               "variants": G.VARIANTS, "anchor_sweep": MAH_ANCHOR_SWEEP,
               "active_aspects": active, "seeds": SEEDS},
              open(MAH_RESULTS_JSON, "w"), indent=2)
    print(f"[save] {MAH_RESULTS_JSON}")

    # ============================================================
    #  v5 EXPERIMENT — RELATIONAL HETEROGENEOUS Graph-JEPA (Ideas 1+2 [+3])
    #  Attacks the SOURCE: builds heterogeneous relational sub-graph targets.
    # ============================================================
    print("\n" + "#" * 124)
    print("#  v5 NEW OBJECTIVE: RELATIONAL HETEROGENEOUS Graph-JEPA ('relhetero')  —  Ideas 1+2 (+3 whitening)")
    print("#  sub-graph target = LN( own_aspect + cross_aspect_neighbors + cited_paper_aspect )")
    print(f"#  WHITEN_INPUTS={WHITEN_INPUTS} (Idea 3)  |  REL_MAX_NEIGH={REL_MAX_NEIGH}")
    print("#" * 124)

    # build the relational/heterogeneous neighbor index ONCE (CPU tensors)
    rel_neighbors = build_hetero_neighbors(data, active, max_neigh=REL_MAX_NEIGH)
    own_r, cross_r, cite_r = rel_neighbors
    print(f"[relhetero] citation pairs (undirected): {cite_r.size(1)}")
    for a in active:
        nn, no, specs = cross_r[a]
        print(f"[relhetero] aspect {a:8s}: cross-aspect neighbors={nn.numel()} "
              f"via {[s[0] for s in specs]}")

    # Idea 3: precompute whitened features once (optional)
    whitened_x = whiten_features(data, active) if WHITEN_INPUTS else None
    if whitened_x is not None:
        for k in whitened_x:
            print(f"[relhetero] whitened '{k}': eff_rank {eff_rank_only(whitened_x[k]):.1f} "
                  f"/ {whitened_x[k].size(1)} (raw was low if this jumps up)")

    rel_cells = []
    # We test relhetero with and without whitening so the ablation is in the table.
    REL_CONFIGS = [("relhetero+white", True), ("relhetero", False)] if WHITEN_INPUTS \
                  else [("relhetero", False)]
    for name in G.VARIANTS:
        for cfg_tag, use_white in REL_CONFIGS:
            wx = whitened_x if use_white else None
            try:
                c = run_cell_multi(name, REL_POOL, data, active, patch_idx, pres, label,
                                   rwse, maskable, pe_dim, rel_neighbors=rel_neighbors,
                                   whitened_x=wx, tag_extra=("WHITEN" if use_white else "raw"))
                c["rel_config"] = cfg_tag
                rel_cells.append(c)
            except Exception as e:
                print(f"  [ERROR] relhetero cell {name} x {cfg_tag} failed: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ---- RELHETERO COMPARISON TABLE (side-by-side with collapsed mean baseline) ----
    print("\n" + "=" * 130)
    print("  RELHETERO COMPARISON  (mean ± std / %d seeds)  —  does a heterogeneous/relational TARGET beat rank-2?" % len(SEEDS))
    print("=" * 130)
    print(f"  {'variant':<26}| {'method':<20}| {'A1 acc':<11}| {'MRR':<11}| {'H@10':<7}| "
          f"{'pm_rk':<9}| {'node_rk':<9}| verdict")
    print("  " + "-" * 126)
    for name in G.VARIANTS:
        base = mean_baseline.get(name)
        if base is not None:
            print(f"  {name:<26}| {'orig mean (own only)':<20}| "
                  f"{base['A1_acc'][0]:.3f}±{base['A1_acc'][1]:.2f}| "
                  f"{base['MRR'][0]:.3f}±{base['MRR'][1]:.2f}| {base['Hits@10'][0]:<7.3f}| "
                  f"{base['patchmean_eff_rank'][0]:<9.1f}| {base['node_eff_rank'][0]:<9.1f}| {base['verdict']}")
        for c in [c for c in rel_cells if c["variant"] == name]:
            print(f"  {name:<26}| {c.get('rel_config','relhetero'):<20}| "
                  f"{c['A1_acc'][0]:.3f}±{c['A1_acc'][1]:.2f}| "
                  f"{c['MRR'][0]:.3f}±{c['MRR'][1]:.2f}| {c['Hits@10'][0]:<7.3f}| "
                  f"{c['patchmean_eff_rank'][0]:<9.1f}| {c['node_eff_rank'][0]:<9.1f}| {c['verdict']}")
        print("  " + "-" * 126)
    print("=" * 130)
    print("  READING (relhetero): the sub-graph target now mixes DIFFERENT node types + CITED papers.")
    print("  REAL FIX  = pm_rk rises WELL above 2 AND MRR rises WELL above chance AND A1 stays ~0.85.")
    print("  If pm_rk rises but MRR flat -> rank restored but task still hard (still a clean finding).")
    print("  If still pm_rk~2 -> collapse is even deeper (inputs fundamentally low-rank).")
    print("=" * 130)

    json.dump({
        "method": "NEW #4 (Ideas 1+2+3): relhetero — relational heterogeneous sub-graph target. "
                  "sub-graph = LN(own_aspect + cross_aspect_reasoning_neighbors + cited_paper_aspect). "
                  "Optional PCA-whitening of raw features (Idea 3). Pure JEPA.",
        "rel_cells": rel_cells, "mean_baseline_cells":
            [c for c in cells if c["pool_mode"] == "mean"],
        "variants": G.VARIANTS, "whiten_inputs": WHITEN_INPUTS, "rel_max_neigh": REL_MAX_NEIGH,
        "rel_cross_edges": {k: [list(t) for t in v] for k, v in REL_CROSS_EDGES.items()},
        "active_aspects": active, "seeds": SEEDS,
        "maskable_papers": int(maskable.numel()), "total_papers": int(P),
        "hypothesis": "The rank-2 collapse is UPSTREAM: the sub-graph target is near-1D because "
                      "it averages a paper's OWN near-duplicate/singleton aspect nodes. Building "
                      "the target from HETEROGENEOUS (different node types) + RELATIONAL (cited "
                      "papers) sources injects paper-specific structural variance that averaging "
                      "cannot destroy, which SHOULD raise sub-graph eff_rank and MRR.",
    }, open(REL_RESULTS_JSON, "w"), indent=2)
    print(f"[save] {REL_RESULTS_JSON}")

    # ============================================================
    #  v6 EXPERIMENT — ORACLE RETRIEVAL TEST (NO TRAINING)
    #  Is MRR~chance a TASK problem (unidentifiable) or PIPELINE?
    # ============================================================
    print("\n" + "#" * 124)
    print("#  v6 ORACLE RETRIEVAL TEST  —  NO training, NO GNN, NO JEPA. Pure input-feature retrieval.")
    print("#  ctx  = retrieve masked aspect from MEAN of the paper's OTHER aspects (the task's ceiling)")
    print("#  cheat= retrieve masked aspect from ITS OWN embedding (sanity: metric works if this is ~1.0)")
    print("#" * 124)

    oracle_all = {}
    for use_white, tag in [(True, "whitened"), (False, "raw")]:
        wx = whitened_x if (use_white and whitened_x is not None) else None
        try:
            res = oracle_retrieval_test(data, active, patch_idx, pres, maskable,
                                        whitened_x=wx, use_white=use_white,
                                        sample_q=ORACLE_SAMPLE_Q, seed=0)
        except Exception as e:
            print(f"  [ERROR] oracle ({tag}) failed: {e}")
            res = {}
        oracle_all[tag] = res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 124)
    print("  ORACLE RETRIEVAL  —  training-free upper bound on the patch-retrieval task")
    print("=" * 124)
    print(f"  {'features':<10}| {'aspect':<8}| {'ctx_MRR':<9}| {'ctx_H@10':<9}| "
          f"{'cheat_MRR':<10}| {'cheat_H@10':<11}| {'N_query':<8}| ~chance")
    print("  " + "-" * 120)
    for tag in ["whitened", "raw"]:
        for a in active:
            r = oracle_all.get(tag, {}).get(a, {})
            has_a = int(pres[:, active.index(a)].sum().item())
            chance = 1.0 / max(1, has_a)
            print(f"  {tag:<10}| {a:<8}| "
                  f"{r.get('ctx_MRR', float('nan')):<9.3f}| {r.get('ctx_H@10', float('nan')):<9.3f}| "
                  f"{r.get('cheat_MRR', float('nan')):<10.3f}| {r.get('cheat_H@10', float('nan')):<11.3f}| "
                  f"{r.get('N', 0):<8}| {chance:.5f}")
        print("  " + "-" * 120)
    print("=" * 124)
    print("  HOW TO READ THIS TABLE (the decisive diagnostic):")
    print("   * cheat_MRR ~ 1.0   => the retrieval metric/machinery is SOUND (expected).")
    print("   * ctx_MRR  >> chance => the masked aspect IS recoverable from other aspects")
    print("                           -> your JEPA/eval PIPELINE is losing it (FIXABLE).")
    print("   * ctx_MRR  ~ chance  => the masked aspect is NOT identifiable from context")
    print("                           -> the TASK is unwinnable as posed (REFRAME the task).")
    print("   * whitened vs raw ctx_MRR shows whether de-anisotropy helps identifiability.")
    print("=" * 124)

    json.dump({
        "method": "NEW #5 (v6): training-free ORACLE retrieval test. Builds patch reprs "
                  "directly from raw/whitened input embeddings (no GNN, no JEPA). 'ctx' = "
                  "retrieve masked aspect from mean of the paper's OTHER aspects (task ceiling); "
                  "'cheat' = retrieve from the target's own embedding (metric sanity).",
        "oracle": oracle_all, "active_aspects": active, "seeds": [0],
        "sample_q": ORACLE_SAMPLE_Q, "whiten_inputs": WHITEN_INPUTS,
        "interpretation": "ctx_MRR>>chance => pipeline-fixable; ctx_MRR~chance => task-unidentifiable.",
    }, open(ORACLE_RESULTS_JSON, "w"), indent=2)
    print(f"[save] {ORACLE_RESULTS_JSON}")


if __name__ == "__main__":
    main()