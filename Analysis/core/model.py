import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter
from einops.layers.torch import Rearrange
import core.model_utils.gMHA_wrapper as gMHA_wrapper

from core.model_utils.elements import MLP
from core.model_utils.feature_encoder import FeatureEncoder
from core.model_utils.gnn import GNN


def variance_loss(z, gamma=1.0, eps=1e-4):
    """VICReg variance hinge: pushes per-dim std toward gamma. z: (N, D)."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z):
    """VICReg covariance: pushes off-diagonal cov toward 0. z: (N, D)."""
    n, d = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (n - 1)
    off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
    return off_diag.pow(2).sum() / d


class GraphJepa(nn.Module):

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 node_type, edge_type,
                 gnn_type,
                 gMHA_type='MLPMixer',
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 patch_rw_dim=0,
                 num_context_patches=1,
                 num_target_patches=4,
                 ema_momentum=0.996):

        super().__init__()
        self.dropout = dropout
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.n_patches = n_patches
        self.pooling = pooling
        self.res = res
        self.patch_rw_dim = patch_rw_dim
        self.nhid = nhid
        self.nfeat_edge = nfeat_edge
        self.num_context_patches = num_context_patches
        self.num_target_patches = num_target_patches
        self.ema_momentum = ema_momentum  # NEW

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(self.patch_rw_dim, nhid, 1)

        # ─────────────────────────────────────────────────────────────
        # FIX: the backbone (input_encoder, edge_encoder, gnns, U, patch_rw
        # encoder) must exist as an ONLINE version AND a frozen EMA TARGET
        # version. Otherwise context & target share one trainable backbone and
        # the model collapses (both branches read the same subgraph_x).
        # We build the online backbone here; the target backbone is an EMA copy.
        # ─────────────────────────────────────────────────────────────
        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)
        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                       bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn - 1)])

        self.reshape = Rearrange('(B p) d ->  B p d', p=n_patches)

        self.context_encoder = getattr(gMHA_wrapper, gMHA_type)(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)
        self.target_encoder = getattr(gMHA_wrapper, gMHA_type)(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        # Predictor: context -> target embedding (nhid dims).
        self.target_predictor = MLP(
            nhid, nhid, nlayer=3, with_final_activation=False, with_norm=True)

        # ── EMA TARGET BACKBONE (frozen; updated by EMA, never by grad) ──
        self.target_input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.target_edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)
        self.target_gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                              bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.target_U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn - 1)])
        if self.patch_rw_dim > 0:
            self.target_patch_rw_encoder = MLP(self.patch_rw_dim, nhid, 1)

        # Initialize ALL target modules = online copies, and freeze them.
        self._init_target()

    # ─────────────────────────────────────────────────────────────
    def _online_target_pairs(self):
        """(online_module, target_module) pairs kept in EMA sync."""
        pairs = [
            (self.input_encoder, self.target_input_encoder),
            (self.edge_encoder, self.target_edge_encoder),
            (self.gnns, self.target_gnns),
            (self.U, self.target_U),
            (self.context_encoder, self.target_encoder),
        ]
        if self.patch_rw_dim > 0:
            pairs.append((self.patch_rw_encoder, self.target_patch_rw_encoder))
        return pairs

    @torch.no_grad()
    def _init_target(self):
        for online, target in self._online_target_pairs():
            target.load_state_dict(online.state_dict())
            for p in target.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def update_target(self, momentum=None):
        """Call this AFTER optimizer.step() every training iteration."""
        m = self.ema_momentum if momentum is None else momentum
        for online, target in self._online_target_pairs():
            for p_o, p_t in zip(online.parameters(), target.parameters()):
                p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)
            for b_o, b_t in zip(online.buffers(), target.buffers()):
                b_t.data.copy_(b_o.data)  # sync BN running stats

    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _match_adj(data, batch, seq_len, device):
        if hasattr(data, 'coarsen_adj') and data.coarsen_adj is not None \
                and data.coarsen_adj.dim() == 3 \
                and data.coarsen_adj.shape[0] == batch \
                and data.coarsen_adj.shape[-1] == seq_len:
            return data.coarsen_adj.to(device)
        return torch.ones(batch, seq_len, seq_len, device=device)

    def _run_backbone(self, data, patch_pes_reduce, gnns, U, input_encoder,
                      edge_encoder, patch_rw_encoder):
        x = input_encoder(data.x.float()).squeeze()
        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)).float().unsqueeze(-1)
        else:
            edge_attr = edge_attr.float()
        edge_attr = edge_encoder(edge_attr)

        x = x[data.subgraphs_nodes_mapper]
        edge_index = data.combined_subgraphs
        e = edge_attr[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch
        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        patch_pes = scatter(pes, batch_x, dim=0, reduce=patch_pes_reduce)

        for i, gnn in enumerate(gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
                x = x + U[i - 1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)

        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        return subgraph_x, patch_pes

    def _online_backbone(self, data, patch_pes_reduce):
        return self._run_backbone(data, patch_pes_reduce, self.gnns, self.U,
                                  self.input_encoder, self.edge_encoder,
                                  self.patch_rw_encoder)

    @torch.no_grad()
    def _target_backbone(self, data, patch_pes_reduce):
        return self._run_backbone(data, patch_pes_reduce, self.target_gnns,
                                  self.target_U, self.target_input_encoder,
                                  self.target_edge_encoder,
                                  self.target_patch_rw_encoder)

    def _representation(self, data):
        subgraph_x, patch_pes = self._online_backbone(data, patch_pes_reduce='mean')
        subgraph_x = subgraph_x + self.patch_rw_encoder(patch_pes)
        mixer_x = subgraph_x.reshape(len(data.call_n_patches), data.call_n_patches[0][0], -1)
        n_p = mixer_x.shape[1]
        rep_adj = self._match_adj(data, mixer_x.shape[0], n_p, mixer_x.device)
        mixer_x = self.context_encoder(mixer_x, rep_adj, ~data.mask)
        out = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True)
        return out

    def forward(self, data, return_repr=False):
        batch_indexer = torch.tensor(np.cumsum(data.call_n_patches))
        batch_indexer = torch.hstack((torch.tensor(0), batch_indexer[:-1])).to(data.y.device)
        context_subgraph_idx = data.context_subgraph_idx + batch_indexer
        target_subgraphs_idx = torch.vstack(
            [torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(data.y.device)
        target_subgraphs_idx += batch_indexer.unsqueeze(1)

        # ── ONLINE branch (context) — trainable ──
        subgraph_x, patch_pes = self._online_backbone(data, patch_pes_reduce='mean')
        context_subgraphs = subgraph_x[context_subgraph_idx]
        context_pe = patch_pes[context_subgraph_idx]
        context_subgraphs = context_subgraphs + self.patch_rw_encoder(context_pe)
        context_x = context_subgraphs.unsqueeze(1)
        context_mask = data.mask.flatten()[context_subgraph_idx].reshape(-1, self.num_context_patches)
        context_adj = torch.ones(
            context_x.shape[0], self.num_context_patches, self.num_context_patches,
            device=context_x.device)
        context_x = self.context_encoder(context_x, context_adj, ~context_mask)

        # ── TARGET branch — EMA backbone + EMA encoder, NO gradient at all ──
        with torch.no_grad():
            t_subgraph_x, t_patch_pes = self._target_backbone(data, patch_pes_reduce='mean')
            target_subgraphs = t_subgraph_x[target_subgraphs_idx.flatten()]
            target_pes = t_patch_pes[target_subgraphs_idx.flatten()]
            target_x = target_subgraphs.reshape(-1, self.num_target_patches, self.nhid)
            encoded_tpatch_pes = self.target_patch_rw_encoder(target_pes)

            if hasattr(data, 'coarsen_adj') and data.coarsen_adj is not None:
                dev = data.coarsen_adj.device
                subgraph_incides = torch.vstack(
                    [torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(dev)
                patch_adj = data.coarsen_adj[
                    torch.arange(target_x.shape[0], device=dev).unsqueeze(1).unsqueeze(2),
                    subgraph_incides.unsqueeze(1),
                    subgraph_incides.unsqueeze(2)
                ].to(target_x.device)
                target_x = self.target_encoder(target_x, patch_adj, None)
            else:
                tgt_adj = torch.ones(
                    target_x.shape[0], self.num_target_patches, self.num_target_patches,
                    device=target_x.device)
                target_x = self.target_encoder(target_x, tgt_adj, None)
            target_x = F.normalize(target_x, dim=-1)   # (B, T, nhid)

        # predictor uses ONLINE context + target positional encodings
        target_prediction_embeddings = context_x + encoded_tpatch_pes.reshape(
            -1, self.num_target_patches, self.nhid)
        B, T, C = target_prediction_embeddings.shape
        flat = target_prediction_embeddings.reshape(B * T, C)
        target_y = self.target_predictor(flat)
        target_y = target_y.reshape(B, T, -1)
        target_y = F.normalize(target_y, dim=-1)

        if return_repr:
            repr_z = self._representation(data)
            return target_x, target_y, repr_z
        return target_x, target_y

    def encode(self, data):
        # Eval uses the EMA target backbone + target encoder (the stable network).
        with torch.no_grad():
            subgraph_x, patch_pes = self._target_backbone(data, patch_pes_reduce='mean')
            subgraph_x = subgraph_x + self.target_patch_rw_encoder(patch_pes)
            mixer_x = subgraph_x.reshape(len(data.call_n_patches), data.call_n_patches[0][0], -1)
            n_p = mixer_x.shape[1]
            enc_adj = self._match_adj(data, mixer_x.shape[0], n_p, mixer_x.device)
            mixer_x = self.target_encoder(mixer_x, enc_adj, ~data.mask)
            out = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True)
        return out

    def encode_nopool(self, data):
        with torch.no_grad():
            subgraph_x, patch_pes = self._target_backbone(data, patch_pes_reduce='mean')
            subgraph_x = subgraph_x + self.target_patch_rw_encoder(patch_pes)
            mixer_x = subgraph_x.reshape(len(data.call_n_patches), data.call_n_patches[0], -1)
            n_p = mixer_x.shape[1]
            enc_adj = self._match_adj(data, mixer_x.shape[0], n_p, mixer_x.device)
            mixer_x = self.target_encoder(mixer_x, enc_adj, ~data.mask)
        return mixer_x