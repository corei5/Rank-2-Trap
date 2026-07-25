import os
import torch
import random
import time
import numpy as np
from core.log import config_logger
from core.asam import ASAM
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             balanced_accuracy_score, f1_score)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(cfg, model, tag, extra=None):
    """Save a trained Graph-JEPA model checkpoint. Returns the saved path."""
    ckpt_dir = os.path.join('checkpoints', str(cfg.dataset))
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'graphjepa_{tag}.pt')
    payload = {'model': model.state_dict(), 'cfg': cfg}
    if extra is not None:
        payload.update(extra)
    torch.save(payload, ckpt_path)
    print(f'>>> saved trained model to {ckpt_path}')
    return ckpt_path


# ══════════════════════════════════════════════════════════════════════════
#  COLLAPSE DIAGNOSTICS (label-free; labels used only for post-hoc analysis)
# ══════════════════════════════════════════════════════════════════════════
def _effective_rank(X, eps=1e-12):
    Xc = X - X.mean(0, keepdims=True)
    try:
        s = np.linalg.svd(Xc, compute_uv=False)
    except np.linalg.LinAlgError:
        return float('nan')
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * np.log(p + eps)).sum()
    return float(np.exp(entropy))


def _per_group_std(X, groups):
    out = {}
    for g in np.unique(groups):
        Xg = X[groups == g]
        if Xg.shape[0] < 2:
            out[g] = float('nan')
            continue
        out[g] = float(np.sqrt(Xg.var(0) + 1e-12).mean())
    return out


def collapse_diagnostics(X, y, n_clusters=None, tag=''):
    """Label-free imbalance-induced-collapse diagnostic."""
    N, D = X.shape
    metrics = {}

    global_std = float(np.sqrt(X.var(0) + 1e-12).mean())
    dead_dims = int((X.std(0) < 1e-3).sum())
    eff_rank = _effective_rank(X)
    metrics.update(dict(global_std=global_std, dead_dims=dead_dims,
                        eff_rank=eff_rank, n_dims=D))

    # ★ HARD COLLAPSE GUARD: warn loudly if the representation is dead.
    if dead_dims >= 0.9 * D or global_std < 1e-2:
        print(f"  ★★★ WARNING: representation looks COLLAPSED "
              f"(dead_dims={dead_dims}/{D}, global_std={global_std:.5f}). "
              f"Check that train() calls model.update_target() and adds var/cov loss. ★★★")

    if n_clusters is None:
        n_clusters = int(min(max(10, N // 50), 100, N // 2 if N > 2 else 2))
    n_clusters = max(2, min(n_clusters, N - 1))
    try:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        cl = km.fit_predict(X)
        cl_sizes = np.array([(cl == c).sum() for c in range(n_clusters)])
        cl_std = _per_group_std(X, cl)
        stds = np.array([cl_std[c] for c in range(n_clusters)])
        order = np.argsort(cl_sizes)
        k = max(1, n_clusters // 3)
        small_idx, large_idx = order[:k], order[-k:]
        small_std = float(np.nanmean(stds[small_idx]))
        large_std = float(np.nanmean(stds[large_idx]))
        collapse_gap = large_std - small_std
        valid = ~np.isnan(stds)
        if valid.sum() > 2 and np.std(cl_sizes[valid]) > 0 and np.std(stds[valid]) > 0:
            size_std_corr = float(np.corrcoef(np.log(cl_sizes[valid] + 1), stds[valid])[0, 1])
        else:
            size_std_corr = float('nan')
        metrics.update(dict(cluster_small_std=small_std, cluster_large_std=large_std,
                            cluster_collapse_gap=collapse_gap,
                            size_std_corr=size_std_corr, n_clusters=n_clusters))
    except Exception:
        metrics.update(dict(cluster_collapse_gap=float('nan'),
                            size_std_corr=float('nan'), n_clusters=n_clusters))

    if y is not None and len(np.unique(y)) > 1:
        classes, counts = np.unique(y, return_counts=True)
        cls_std = _per_group_std(X, y)
        cstds = np.array([cls_std[c] for c in classes])
        order = np.argsort(counts)
        k = max(1, len(classes) // 3)
        rare_std = float(np.nanmean(cstds[order[:k]]))
        freq_std = float(np.nanmean(cstds[order[-k:]]))
        metrics.update(dict(label_rare_std=rare_std, label_freq_std=freq_std,
                            label_collapse_gap=freq_std - rare_std,
                            imbalance_ratio=float(counts.max() / max(counts.min(), 1)),
                            n_classes=len(classes)))

    print(f"\n[COLLAPSE-DIAG {tag}]")
    print(f"  global: std={global_std:.4f}  dead_dims={dead_dims}/{D}  "
          f"eff_rank={eff_rank:.2f}/{D}")
    if 'cluster_collapse_gap' in metrics:
        print(f"  LABEL-FREE clusters(k={metrics['n_clusters']}): "
              f"small_std={metrics.get('cluster_small_std', float('nan')):.4f}  "
              f"large_std={metrics.get('cluster_large_std', float('nan')):.4f}  "
              f"collapse_gap={metrics['cluster_collapse_gap']:.4f}  "
              f"(>0 => rare regions collapse more => PHENOMENON PRESENT)")
        print(f"  size~std corr={metrics.get('size_std_corr', float('nan')):.3f} "
              f"(>0 => bigger clusters healthier)")
    if 'label_collapse_gap' in metrics:
        print(f"  LABEL cross-check: imbalance_ratio={metrics['imbalance_ratio']:.1f}  "
              f"rare_std={metrics['label_rare_std']:.4f}  "
              f"freq_std={metrics['label_freq_std']:.4f}  "
              f"label_collapse_gap={metrics['label_collapse_gap']:.4f}")
    return metrics


# ★ helper: read the EMA warmup range from cfg if present, else use defaults.
def _ema_range(cfg):
    lo = getattr(cfg.jepa, 'ema_momentum', 0.99)
    hi = getattr(cfg.jepa, 'ema_momentum_end', 0.999)
    return [float(lo), float(hi)]


def run(cfg, create_dataset, create_model, train, test, evaluator=None):
    if cfg.seed is not None:
        seeds = [cfg.seed]
        cfg.train.runs = 1
    else:
        seeds = [21, 42, 41, 95, 12, 35, 66, 85, 3, 1234]

    writer, logger = config_logger(cfg)

    train_dataset, val_dataset, test_dataset = create_dataset(cfg)

    train_loader = DataLoader(
        train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(
        val_dataset,  cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(
        test_dataset, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)

    train_losses = []
    per_epoch_times = []
    total_times = []
    maes = []
    for run in range(cfg.train.runs):
        set_seed(seeds[run])
        model = create_model(cfg).to(cfg.device)
        print(f"\nNumber of parameters: {count_parameters(model)}")

        if cfg.train.optimizer == 'ASAM':
            sharp = True
            optimizer = torch.optim.SGD(
                model.parameters(), lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.wd)
            minimizer = ASAM(optimizer, model, rho=0.5)
        else:
            sharp = False
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=cfg.train.lr_decay,
                                                               patience=cfg.train.lr_patience,
                                                               verbose=True)

        start_outer = time.time()
        per_epoch_time = []

        best_val = float('inf')
        best_tag = f'best_run{run}_seed{seeds[run]}'

        ipe = len(train_loader)
        ema_params = _ema_range(cfg)   # ★ was hard-coded [0.99, 0.999]
        momentum_scheduler = (ema_params[0] + i*(ema_params[1]-ema_params[0])/(ipe*cfg.train.epochs)
                              for i in range(int(ipe*cfg.train.epochs)+1))
        for epoch in range(cfg.train.epochs):
            start = time.time()
            model.train()
            _, train_loss = train(
                train_loader, model, optimizer if not sharp else minimizer,
                evaluator=evaluator, device=cfg.device, momentum_weight=next(momentum_scheduler),
                sharp=sharp, criterion_type=cfg.jepa.dist)
            model.eval()
            _, val_loss = test(val_loader, model,
                               evaluator=evaluator, device=cfg.device, criterion_type=cfg.jepa.dist)
            _, test_loss = test(test_loader, model,
                                evaluator=evaluator, device=cfg.device, criterion_type=cfg.jepa.dist)

            time_cur_epoch = time.time() - start
            per_epoch_time.append(time_cur_epoch)

            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val: {val_loss:.4f}, Test: {test_loss:.4f} Seconds: {time_cur_epoch:.4f}')

            writer.add_scalar(f'Run{run}/train-loss', train_loss, epoch)
            writer.add_scalar(f'Run{run}/val-loss', val_loss, epoch)

            if val_loss < best_val:
                best_val = val_loss
                save_model(cfg, model, best_tag,
                           extra={'run': run, 'seed': seeds[run],
                                  'epoch': epoch + 1, 'val_loss': float(val_loss)})

            if scheduler is not None:
                scheduler.step(val_loss)

            if not sharp:
                if optimizer.param_groups[0]['lr'] < cfg.train.min_lr:
                    print("!! LR EQUAL TO MIN LR SET.")
                    break

        per_epoch_time = np.mean(per_epoch_time)
        total_time = (time.time()-start_outer)/3600

        model.eval()
        save_model(cfg, model, f'last_run{run}_seed{seeds[run]}',
                   extra={'run': run, 'seed': seeds[run], 'epoch': epoch + 1})

        X_train, y_train = [], []
        X_test, y_test = [], []
        for data in train_loader:
            data.to(cfg.device)
            with torch.no_grad():
                features = model.encode(data)
                X_train.append(features.detach().cpu().numpy())
                y_train.append(data.y.detach().cpu().numpy())
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        for data in test_loader:
            data.to(cfg.device)
            with torch.no_grad():
                features = model.encode(data)
                X_test.append(features.detach().cpu().numpy())
                y_test.append(data.y.detach().cpu().numpy())
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        print("Data shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        collapse_diagnostics(X_test, None, tag=f'run{run}-test')

        lin_model = Ridge()
        lin_model.fit(X_train, y_train)
        lin_predictions = lin_model.predict(X_test)
        lin_mae = mean_absolute_error(y_test, lin_predictions)
        maes.append(lin_mae)

        print("\nRun: ", run)
        print("Train Loss: {:.4f}".format(train_loss))
        print("Convergence Time (Epochs): {}".format(epoch+1))
        print("AVG TIME PER EPOCH: {:.4f} s".format(per_epoch_time))
        print("TOTAL TIME TAKEN: {:.4f} h".format(total_time))
        print(f'Train R2.: {lin_model.score(X_train, y_train)}')
        print(f'MAE.: {lin_mae}')

        train_losses.append(train_loss)
        per_epoch_times.append(per_epoch_time)
        total_times.append(total_time)

    if cfg.train.runs > 1:
        train_loss = torch.tensor(train_losses)
        per_epoch_time = torch.tensor(per_epoch_times)
        total_time = torch.tensor(total_times)
        print(f'\nFinal Train Loss: {train_loss.mean():.4f} ± {train_loss.std():.4f}'
              f'\nSeconds/epoch: {per_epoch_time.mean():.4f}'
              f'\nHours/total: {total_time.mean():.4f}')
        logger.info("-"*50)
        logger.info(cfg)
        logger.info(f'\nFinal Train Loss: {train_loss.mean():.4f} ± {train_loss.std():.4f}'
                    f'\nSeconds/epoch: {per_epoch_time.mean():.4f}'
                    f'\nHours/total: {total_time.mean():.4f}')
        maes = np.array(maes)
        print(f'MAE avg: {maes.mean()}, std: {maes.std()}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def k_fold(dataset, folds=10):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)
    train_indices, test_indices = [], []
    ys = dataset.data.y
    for train, test in skf.split(torch.zeros(len(dataset)), ys):
        train_indices.append(torch.from_numpy(train).to(torch.long))
        test_indices.append(torch.from_numpy(test).to(torch.long))
    return train_indices, test_indices


def run_k_fold(cfg, create_dataset, create_model, train, test, evaluator=None, k=10):
    if cfg.seed is not None:
        seeds = [cfg.seed]
        cfg.train.runs = 1
    else:
        seeds = [42, 21, 95, 12, 35]

    writer, logger = config_logger(cfg)
    dataset, transform, transform_eval = create_dataset(cfg)

    if hasattr(dataset, 'train_indices'):
        k_fold_indices = dataset.train_indices, dataset.test_indices
    else:
        k_fold_indices = k_fold(dataset, cfg.k)

    train_losses = []
    per_epoch_times = []
    total_times = []
    run_metrics = []
    best_overall_bal = -1.0
    diag_records = []
    for run in range(cfg.train.runs):
        set_seed(seeds[run])
        bal_accs, macro_f1s, raw_accs = [], [], []
        for fold, (train_idx, test_idx) in enumerate(zip(*k_fold_indices)):
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]
            train_dataset.transform = transform
            test_dataset.transform = transform_eval
            test_dataset = [x for x in test_dataset]

            if not cfg.metis.online:
                train_dataset = [x for x in train_dataset]

            train_loader = DataLoader(
                train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
            test_loader = DataLoader(
                test_dataset,  cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)

            model = create_model(cfg).to(cfg.device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=cfg.train.lr_decay,
                                                                   patience=cfg.train.lr_patience,
                                                                   verbose=True)

            start_outer = time.time()
            per_epoch_time = []

            ipe = len(train_loader)
            ema_params = _ema_range(cfg)   # ★ was hard-coded [0.99, 0.999]
            momentum_scheduler = (ema_params[0] + i*(ema_params[1]-ema_params[0])/(ipe*cfg.train.epochs)
                                  for i in range(int(ipe*cfg.train.epochs)+1))

            for epoch in range(cfg.train.epochs):
                start = time.time()
                model.train()
                _, train_loss = train(
                    train_loader, model, optimizer,
                    evaluator=evaluator, device=cfg.device,
                    momentum_weight=next(momentum_scheduler), criterion_type=cfg.jepa.dist)
                model.eval()
                _, test_loss = test(
                    test_loader, model, evaluator=evaluator, device=cfg.device,
                    criterion_type=cfg.jepa.dist)

                scheduler.step(test_loss)
                time_cur_epoch = time.time() - start
                per_epoch_time.append(time_cur_epoch)

                print(f'Epoch/Fold: {epoch:03d}/{fold}, Train Loss: {train_loss:.4f}'
                      f' Test Loss:{test_loss:.4f}, Seconds: {time_cur_epoch:.4f}, ')
                writer.add_scalar(f'Run{run}/train-loss', train_loss, epoch)
                writer.add_scalar(f'Run{run}/test-loss', test_loss, epoch)

                if optimizer.param_groups[0]['lr'] < cfg.train.min_lr:
                    print("!! LR EQUAL TO MIN LR SET.")
                    break

            per_epoch_time = np.mean(per_epoch_time)
            total_time = (time.time()-start_outer)/3600

            model.eval()
            fold_path = save_model(
                cfg, model, f'run{run}_fold{fold}_seed{seeds[run]}',
                extra={'run': run, 'fold': fold, 'seed': seeds[run], 'epoch': epoch + 1})

            X_train, y_train = [], []
            X_test, y_test = [], []
            for data in train_loader:
                data.to(cfg.device)
                with torch.no_grad():
                    features = model.encode(data)
                    X_train.append(features.detach().cpu().numpy())
                    y_train.append(data.y.detach().cpu().numpy())
            X_train = np.concatenate(X_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)

            for data in test_loader:
                data.to(cfg.device)
                with torch.no_grad():
                    features = model.encode(data)
                    X_test.append(features.detach().cpu().numpy())
                    y_test.append(data.y.detach().cpu().numpy())
            X_test = np.concatenate(X_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)

            print("Data shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

            diag = collapse_diagnostics(X_test, y_test, tag=f'run{run}-fold{fold}')
            diag_records.append(diag)

            lin_model = LogisticRegression(
                max_iter=10000, class_weight='balanced', n_jobs=-1)
            lin_model.fit(X_train, y_train)
            lin_predictions = lin_model.predict(X_test)

            raw_acc = accuracy_score(y_test, lin_predictions)
            bal_acc = balanced_accuracy_score(y_test, lin_predictions)
            macro_f1 = f1_score(y_test, lin_predictions, average='macro', zero_division=0)
            weighted_f1 = f1_score(y_test, lin_predictions, average='weighted', zero_division=0)

            raw_accs.append(raw_acc)
            bal_accs.append(bal_acc)
            macro_f1s.append(macro_f1)

            print(f'[PROBE] raw_acc={raw_acc:.4f} | balanced_acc={bal_acc:.4f} '
                  f'| macro_F1={macro_f1:.4f} | weighted_F1={weighted_f1:.4f}')

            if bal_acc > best_overall_bal:
                best_overall_bal = bal_acc
                save_model(cfg, model, 'best',
                           extra={'run': run, 'fold': fold, 'seed': seeds[run],
                                  'epoch': epoch + 1, 'balanced_acc': float(bal_acc),
                                  'raw_acc': float(raw_acc), 'macro_f1': float(macro_f1)})

            print(f'Fold {fold}, Seconds/epoch: {per_epoch_time}')
            train_losses.append(train_loss)
            per_epoch_times.append(per_epoch_time)
            total_times.append(total_time)

        print("\nRun: ", run)
        print("Train Loss: {:.4f}".format(train_loss))
        print("Convergence Time (Epochs): {}".format(epoch+1))
        print("AVG TIME PER EPOCH: {:.4f} s".format(per_epoch_time))
        print("TOTAL TIME TAKEN: {:.4f} h".format(total_time))
        bal_accs = np.array(bal_accs); macro_f1s = np.array(macro_f1s); raw_accs = np.array(raw_accs)
        print(f'Balanced Acc: {bal_accs.mean():.4f} ± {bal_accs.std():.4f} '
              f'| Macro-F1: {macro_f1s.mean():.4f} ± {macro_f1s.std():.4f} '
              f'| Raw Acc: {raw_accs.mean():.4f} ± {raw_accs.std():.4f}')
        run_metrics.append([bal_accs.mean(), bal_accs.std(), macro_f1s.mean(), raw_accs.mean()])
        print()

    if cfg.train.runs > 1:
        train_loss = torch.tensor(train_losses)
        per_epoch_time = torch.tensor(per_epoch_times)
        total_time = torch.tensor(total_times)
        print(f'\nFinal Train Loss: {train_loss.mean():.4f} ± {train_loss.std():.4f}'
              f'\nSeconds/epoch: {per_epoch_time.mean():.4f}'
              f'\nHours/total: {total_time.mean():.4f}')
        logger.info("-"*50)
        logger.info(cfg)
        logger.info(f'\nFinal Train Loss: {train_loss.mean():.4f} ± {train_loss.std():.4f}'
                    f'\nSeconds/epoch: {per_epoch_time.mean():.4f}'
                    f'\nHours/total: {total_time.mean():.4f}')

    run_metrics = np.array(run_metrics)
    print('Averages over runs (balanced_acc, bal_std, macro_f1, raw_acc):')
    print(run_metrics.mean(axis=0))

    if diag_records:
        cg = np.array([d.get('cluster_collapse_gap', np.nan) for d in diag_records])
        lg = np.array([d.get('label_collapse_gap', np.nan) for d in diag_records])
        corr = np.array([d.get('size_std_corr', np.nan) for d in diag_records])
        ir = np.array([d.get('imbalance_ratio', np.nan) for d in diag_records])
        print("\n" + "="*60)
        print("IMBALANCE-INDUCED-COLLAPSE SUMMARY (label-free)")
        print(f"  imbalance_ratio (mean)      : {np.nanmean(ir):.1f}")
        print(f"  cluster_collapse_gap (mean) : {np.nanmean(cg):.4f}  "
              f"(>0 => rare regions collapse more)")
        print(f"  size~std corr (mean)        : {np.nanmean(corr):.3f}")
        print(f"  label_collapse_gap (mean)   : {np.nanmean(lg):.4f}  "
              f"(analysis-only cross-check)")
        if np.nanmean(cg) > 0.02 and np.nanmean(corr) > 0.15:
            print("  >>> VERDICT: phenomenon PRESENT — CB-VCR is well motivated. Proceed.")
        else:
            print("  >>> VERDICT: phenomenon WEAK — reconsider framing before writing method.")
        print("="*60)

    print(f'\n>>> Best model (balanced_acc={best_overall_bal:.4f}) saved as '
          f'checkpoints/{cfg.dataset}/graphjepa_best.pt')