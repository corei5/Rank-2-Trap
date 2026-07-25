import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

BN = True


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass


class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=BN, bias=True):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList([nn.Linear(nin if i == 0 else n_hid,
                                     n_hid if i < nlayer-1 else nout,
                                     # TODO: revise later
                                               bias=True if (i == nlayer-1 and not with_final_activation and bias)
                                               or (not with_norm) else False)  # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i < nlayer-1 else nout) if with_norm else Identity()
                                    for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin == nout)  # TODO: test whether need this

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        # ─────────────────────────────────────────────────────────────
        # SAFETY NET: BatchNorm1d expects 2-D (N, C). If a 3-D (B, T, C)
        # tensor is passed, flatten to (B*T, C) for the layer stack and
        # restore the original shape at the end. (Callers in model.py now
        # pass 2-D, so this is only a guard for any other 3-D call site.)
        # Remove this block if you want the file identical to the original.
        # ─────────────────────────────────────────────────────────────
        orig_shape = None
        if x.dim() == 3:
            orig_shape = x.shape
            x = x.reshape(-1, x.shape[-1])

        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = F.relu(x)

        # if self.residual:
        #     x = x + previous_x

        if orig_shape is not None:
            x = x.reshape(orig_shape[0], orig_shape[1], -1)
        return x


class VNUpdate(nn.Module):
    def __init__(self, dim, with_norm=BN):
        """
        Intermediate update layer for the virtual node
        :param dim: Dimension of the latent node embeddings
        :param config: Python Dict with the configuration of the CRaWl network
        """
        super().__init__()
        self.mlp = MLP(dim, dim, with_norm=with_norm,
                       with_final_activation=True, bias=not BN)

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, vn, x, batch):
        G = global_add_pool(x, batch)
        if vn is not None:
            G += vn
        vn = self.mlp(G)
        x += vn[batch]
        return vn, x