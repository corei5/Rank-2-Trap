import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader

from core.config import cfg, update_cfg
from core.get_data import create_dataset
from core.get_model import create_model


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs, ys = [], []
    for data in loader:
        data = data.to(device)
        z = model.encode(data)          # <-- uses the encode() patch ([0])
        if z.dim() > 2:
            z = z.reshape(z.size(0), -1)
        embs.append(z.cpu().numpy())
        ys.append(data.y.cpu().numpy())
    return np.concatenate(embs), np.concatenate(ys).ravel()


def main():
    cfg.merge_from_file('train/configs/papers.yaml')
    update_cfg(cfg)
    device = torch.device(f'cuda:{cfg.device}' if torch.cuda.is_available() else 'cpu')

    # dataset + transforms
    dataset, transform_train, transform_eval = create_dataset(cfg)
    dataset.transform = transform_eval
    label_map = dataset.label_map
    inv_map = {v: k for k, v in label_map.items()}

    # model
    model = create_model(cfg).to(device)
    ckpt = 'best_papers_model.pt'                 # <- see note below
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f'Loaded checkpoint {ckpt}')
    else:
        print('WARNING: no checkpoint found, using freshly-initialized model.')

    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    X, y = extract_embeddings(model, loader, device)
    print(f'Embeddings: {X.shape}, labels: {y.shape}, classes: {len(set(y))}')

    # ---- probe: train/test split accuracy ----
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                           stratify=y, random_state=0)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f'\n>>> field_subfield probe accuracy: {acc:.4f}\n')

    # ---- 1) t-SNE scatter colored by field ----
    n_show = min(3000, len(X))
    idx = np.random.choice(len(X), n_show, replace=False)
    emb2d = TSNE(n_components=2, init='pca', perplexity=30,
                 random_state=0).fit_transform(X[idx])
    plt.figure(figsize=(11, 9))
    classes = sorted(set(y[idx]))
    cmap = plt.cm.get_cmap('tab20', len(classes))
    for ci, c in enumerate(classes):
        m = y[idx] == c
        plt.scatter(emb2d[m, 0], emb2d[m, 1], s=6, color=cmap(ci),
                    label=inv_map.get(c, str(c)))
    if len(classes) <= 20:
        plt.legend(markerscale=2, fontsize=7, loc='best')
    plt.title('Paper embeddings (t-SNE) colored by field_subfield')
    plt.tight_layout()
    plt.savefig('viz_tsne.png', dpi=150)
    print('saved viz_tsne.png')

    # ---- 2) confusion matrix ----
    labels_sorted = sorted(set(y))
    cm = confusion_matrix(yte, ypred, labels=labels_sorted)
    plt.figure(figsize=(10, 9))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    if len(labels_sorted) <= 25:
        names = [inv_map.get(l, str(l)) for l in labels_sorted]
        plt.xticks(range(len(names)), names, rotation=90, fontsize=6)
        plt.yticks(range(len(names)), names, fontsize=6)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'Confusion matrix (acc={acc:.3f})')
    plt.tight_layout()
    plt.savefig('viz_confusion.png', dpi=150)
    print('saved viz_confusion.png')

    # ---- 3) per-class accuracy bar ----
    per_cls = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    order = np.argsort(per_cls)
    plt.figure(figsize=(11, max(4, len(labels_sorted) * 0.25)))
    names = [inv_map.get(labels_sorted[i], str(labels_sorted[i])) for i in order]
    plt.barh(range(len(order)), per_cls[order], color='steelblue')
    plt.yticks(range(len(order)), names, fontsize=6)
    plt.xlabel('Accuracy'); plt.title('Per-class accuracy')
    plt.tight_layout()
    plt.savefig('viz_per_class.png', dpi=150)
    print('saved viz_per_class.png')


if __name__ == '__main__':
    main()
