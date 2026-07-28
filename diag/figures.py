from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "figure.dpi": 160,
                     "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25})


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p); plt.close(fig)
    print(f"  [fig] {p}")


def render_all(db: dict, out_dir: str, tag: str = ""):
    sfx = f"_{tag}" if tag else ""

    # F1 ridge probe
    if "ridge" in db:
        pr = db["ridge"]["probes"]
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        k = list(pr); v = [pr[i]["mrr"] for i in k]; nl = [pr[i]["null_mrr"] for i in k]
        x = np.arange(len(k))
        ax.bar(x - .2, v, .4, label="ridge probe")
        ax.bar(x + .2, nl, .4, label="permuted null")
        ax.set_yscale("log"); ax.set_xticks(x)
        ax.set_xticklabels([i.replace("_", "\n") for i in k], fontsize=7)
        ax.set_ylabel("MRR"); ax.legend(fontsize=7)
        ax.set_title("Ridge: context $\\to$ oracle retrieval space")
        _save(fig, out_dir, f"figR1_ridge{sfx}.pdf")

    # F2 loss vs floor
    if "lossfloor" in db:
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        for i, r in enumerate(db["lossfloor"]["per_seed"]):
            ax.plot(r["loss_curve_tail"], alpha=.7, label=f"seed {i}" if i < 3 else None)
        fl = db["lossfloor"]["per_seed"][0]["floor"]
        ax.axhline(fl, ls="--", c="k", label=r"$\mathbb{E}\|\delta\|^2$")
        ax.set_xlabel("epoch (tail)"); ax.set_ylabel("loss"); ax.legend(fontsize=7)
        ax.set_title(f"$\\mathcal{{L}}/$floor $= {db['lossfloor']['mean_ratio']:.3f}$")
        _save(fig, out_dir, f"figR2_lossfloor{sfx}.pdf")

    # F3 gradient audit
    if "gradaudit" in db:
        fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
        for m, d in db["gradaudit"]["modules"].items():
            ax[0].plot(d["curve"], label=m)
        ax[0].set_yscale("log"); ax[0].set_xlabel("epoch")
        ax[0].set_ylabel(r"$\|g\|_2$"); ax[0].legend(fontsize=7)
        ax[1].plot(db["gradaudit"]["rank_curve"], "o-")
        ax[1].set_xlabel("probe"); ax[1].set_ylabel("effective rank")
        fig.suptitle("Predictor is training; rank is flat by optimum, not by death",
                     fontsize=8)
        _save(fig, out_dir, f"figR3_gradaudit{sfx}.pdf")

    # F4 positive control
    if "posctrl" in db:
        c = db["posctrl"]["controls"]
        fig, ax = plt.subplots(figsize=(4.6, 2.6))
        k = list(c); b = [c[i]["bits_mean"] for i in k]
        ax.barh(k, b); ax.axvline(0, c="k", lw=.8)
        ax.set_xlabel("bits recovered (0 = chance)")
        ax.set_title("Positive controls, same encoder & corpus")
        _save(fig, out_dir, f"figR4_posctrl{sfx}.pdf")

    # F5 pool ladder
    if "poolladder" in db:
        L = db["poolladder"]["ladder"]
        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        for name, rows in L.items():
            ax.plot([r["K"] for r in rows], [r["bits"] for r in rows], "o-", label=name)
        ax.set_xscale("log"); ax.set_xlabel("hard-pool size $K$")
        ax.set_ylabel("bits recovered"); ax.legend(fontsize=7)
        ax.set_title("Difficulty ladder (replaces every ratio-to-chance)")
        _save(fig, out_dir, f"figR5_ladder{sfx}.pdf")

    # F6 phase diagram
    if "phase" in db:
        g = db["phase"]["grid"]
        rhos = sorted({r["rho"] for r in g}); kaps = sorted({r["kappa"] for r in g})
        M = np.full((len(kaps), len(rhos)), np.nan)
        for r in g:
            M[kaps.index(r["kappa"]), rhos.index(r["rho"])] = r["mrr"]
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                       norm=matplotlib.colors.LogNorm(vmin=max(M.min(), 1e-6),
                                                      vmax=max(M.max(), 1e-5)))
        ax.set_xticks(range(len(rhos)));  ax.set_xticklabels([f"{r:.4f}" for r in rhos],
                                                            rotation=45, fontsize=7)
        ax.set_yticks(range(len(kaps)));  ax.set_yticklabels(kaps, fontsize=7)
        ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$\kappa$ (anisotropy)")
        m = db["phase"]["measured"]
        ax.plot(rhos.index(min(rhos, key=lambda r: abs(r - m["rho"]))),
                min(range(len(kaps)), key=lambda i: abs(kaps[i] - m["kappa"])),
                "r*", ms=14, label="measured")
        ax.legend(fontsize=7); fig.colorbar(im, ax=ax, label="MRR")
        ax.set_title(r"$\rho$ alone is not sufficient")
        _save(fig, out_dir, f"figR6_phase{sfx}.pdf")

    # F7 hard pool
    if "hardpool" in db:
        P = db["hardpool"]["pools"]
        keys = list(P); meth = list(P[keys[0]])
        fig, ax = plt.subplots(figsize=(5.2, 2.8))
        w = .8 / len(meth)
        for i, m in enumerate(meth):
            ax.bar(np.arange(len(keys)) + i * w, [P[k][m]["mrr"] for k in keys],
                   w, label=m)
        ax.set_xticks(np.arange(len(keys)) + .4); ax.set_xticklabels(keys)
        ax.set_ylabel("MRR"); ax.legend(fontsize=7)
        ax.set_title("Hard within-category pools break the lexical shortcut")
        _save(fig, out_dir, f"figR7_hardpool{sfx}.pdf")