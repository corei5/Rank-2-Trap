import torch
import numpy as np
import torch.nn.functional as F
from core.config import cfg, update_cfg
from core.get_data import create_dataset
from core.get_model import create_model
from core.trainer import run_k_fold
from core.model_utils.hyperbolic_dist import hyperbolic_dist


# ─────────────────────────────────────────────────────────────
# JEPA/BYOL prediction loss: 1 - cosine(pred, target).
# pred, target: (B, T, nhid), already L2-normalized in the model.
# ─────────────────────────────────────────────────────────────
def prediction_loss(pred, target):
    cos = (pred * target).sum(dim=-1)      # (B, T)
    return (1.0 - cos).mean()


# ─────────────────────────────────────────────────────────────
# VICReg-style anti-collapse regularizers on the 512-d graph representation.
# ─────────────────────────────────────────────────────────────
def variance_loss(z, gamma=1.0, eps=1e-4):
    """Force every embedding dim to keep std >= gamma."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z):
    """Push off-diagonal covariances toward 0 (decorrelate the 512 dims)."""
    n, d = z.shape
    if n <= 1:
        return z.sum() * 0.0
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (n - 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / d


def train(train_loader, model, optimizer, evaluator, device,
          momentum_weight, sharp=None, criterion_type=0):

    # ─────────────────────────────────────────────────────────────
    # Coefficients. The prediction term is now a PROPER predictive task
    # (online context -> EMA target, separate backbones) so it cannot be
    # solved trivially. var/cov are safety regularizers. sim leads.
    #  - sim_coeff moderate (was 25 which drowned everything)
    #  - var_coeff strong enough to forbid dead dims
    # ─────────────────────────────────────────────────────────────
    sim_coeff, var_coeff, cov_coeff = 1.0, 1.0, 0.04

    model.train()
    step_losses, num_targets = [], []
    first_batch = None
    for data in train_loader:
        if model.use_lap:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)

        data = data.to(device)
        optimizer.zero_grad()

        # target_x = FULL L2-normalized EMA target embedding (B, T, nhid)  [no grad]
        # target_y = FULL L2-normalized predictor output          (B, T, nhid)
        target_x, target_y, repr_z = model(data, return_repr=True)

        # 1) prediction term (full-embedding cosine)
        inv_loss = prediction_loss(target_y, target_x)

        # 2) anti-collapse on the 512-d representation
        var = variance_loss(repr_z)
        cov = covariance_loss(repr_z)

        # DEBUG once per epoch
        if first_batch is None:
            first_batch = data
            with torch.no_grad():
                enc = model.encode(data)               # EMA target encoder = what eval uses
                enc_std = enc.std(0).mean().item()
                enc_dead = int((enc.std(0) < 1e-3).sum().item())
            print(f"[DEBUG] repr_z.std(mean)={repr_z.std(0).mean().item():.4f} "
                  f"| var_loss={var.item():.4f} | cov_loss={cov.item():.4f} "
                  f"| inv_loss={inv_loss.item():.4f} "
                  f"|| encode().std(mean)={enc_std:.4f} dead_dims={enc_dead}/512 "
                  f"| momentum={momentum_weight:.4f}")

        loss = sim_coeff * inv_loss + var_coeff * var + cov_coeff * cov

        step_losses.append(inv_loss.item())   # ★ log the PREDICTION loss (comparable, must be > 0)
        num_targets.append(len(target_y))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ─────────────────────────────────────────────────────────────
        # ★ CRITICAL FIX: EMA-update the ENTIRE target network (backbone +
        # encoder + patch_rw), NOT just target_encoder. The old loop only
        # updated target_encoder while the shared backbone was gradient-trained
        # AND fed both branches -> guaranteed collapse. model.update_target()
        # (defined in the new core/model.py) syncs every online->target module.
        # ─────────────────────────────────────────────────────────────
        model.update_target(momentum_weight)

    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


@torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    model.eval()
    step_losses, num_targets = [], []
    for data in loader:
        data = data.to(device)
        target_x, target_y, repr_z = model(data, return_repr=True)
        inv_loss = prediction_loss(target_y, target_x)
        step_losses.append(inv_loss.item())   # ★ report prediction loss
        num_targets.append(len(target_y))
    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/papers.yaml')
    cfg = update_cfg(cfg)
    cfg.k = 2  # 10
    run_k_fold(cfg, create_dataset, create_model, train, test)