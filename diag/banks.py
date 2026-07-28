"""Protocol R banks + the training-free oracle + BM25, one API.

Rewritten for corpus-scale (n ~ 6e4) use:
  * BM25 is a precomputed sparse weight matrix, built ONCE per candidate list
    and cached, and scores only the documents inside the candidate pool.
  * hard_pool() is chunked (no |category|^2 allocation) and can build pools for
    a subset of queries only.
  * pooled_ranks() sizes its chunk from K so a K=1000 pool cannot OOM.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from typing import Dict, Any, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from diag.common import device


# ============================================================ protocol-R frames
def protocol_r_frame(Q: torch.Tensor, C: torch.Tensor, frame: str = "raw"):
    """raw | center | rm-top | zca. Centring uses candidate-bank statistics only."""
    Q, C = Q.float(), C.float()
    if frame == "raw":
        pass
    elif frame == "center":
        mu = C.mean(0, keepdim=True); Q, C = Q - mu, C - mu
    elif frame == "rm-top":
        mu = C.mean(0, keepdim=True); Q, C = Q - mu, C - mu
        v = torch.linalg.svd(C, full_matrices=False)[2][:1]
        Q = Q - (Q @ v.T) @ v; C = C - (C @ v.T) @ v
    elif frame == "zca":
        mu = C.mean(0, keepdim=True); Qc, Cc = Q - mu, C - mu
        cov = (Cc.T @ Cc) / max(len(Cc) - 1, 1)
        ev, V = torch.linalg.eigh(cov.double())
        W = (V @ torch.diag((ev.clamp(min=1e-6)) ** -0.5) @ V.T).float()
        Q, C = Qc @ W, Cc @ W
    else:
        raise ValueError(frame)
    return F.normalize(Q, dim=-1), F.normalize(C, dim=-1)


def training_free_oracle(X: torch.Tensor, query_aspect: int = 0) -> Dict[str, torch.Tensor]:
    """Mean of the visible aspects as the query; masked aspect as candidate."""
    Z = F.normalize(X.float(), dim=-1)
    A = Z.shape[1]
    vis = [a for a in range(A) if a != query_aspect]
    Q = F.normalize(Z[:, vis, :].mean(1), dim=-1)
    C = Z[:, query_aspect, :]
    return {"Q": Q, "C": C, "gold": torch.arange(len(Z))}


# ==================================================================== lexical
_WORD = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> List[str]:
    return _WORD.findall(s.lower())


class BM25Index:
    """Okapi BM25 as a precomputed CSR weight matrix.

    score(q, d) = sum_{t in q} count_q(t) * W[d, t]
    with W[d, t] = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl)).

    Because W is precomputed, scoring a pool of P documents costs O(nnz(W[pool]))
    instead of O(|q| * n_docs), which is the difference between minutes and days
    at n_docs = 6e4.
    """

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75,
                 verbose: bool = True):
        from scipy.sparse import csr_matrix

        t0 = time.time()
        n = len(docs)
        vocab: Dict[str, int] = {}
        indptr = np.zeros(n + 1, dtype=np.int64)
        indices: List[int] = []
        data: List[float] = []
        dl = np.zeros(n, dtype=np.float64)

        for i, d in enumerate(docs):
            toks = _tok(d)
            dl[i] = len(toks)
            for w, f in Counter(toks).items():
                j = vocab.get(w)
                if j is None:
                    j = len(vocab); vocab[w] = j
                indices.append(j); data.append(float(f))
            indptr[i + 1] = len(indices)

        V = max(len(vocab), 1)
        tf = np.asarray(data, dtype=np.float32)
        ind = np.asarray(indices, dtype=np.int32)
        df = np.bincount(ind, minlength=V).astype(np.float64)      # CSR => df exactly
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)

        avgdl = float(dl.mean()) if n else 1.0
        denom_row = (k1 * (1.0 - b + b * dl / max(avgdl, 1e-9))).astype(np.float32)
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        w = tf * (k1 + 1.0) / (tf + denom_row[rows]) * idf[ind]

        self.W = csr_matrix((w, ind, indptr), shape=(n, V))
        self.vocab = vocab
        self.n_docs = n
        self.n_vocab = V
        self._buf = np.zeros(V, dtype=np.float32)
        if verbose:
            print(f"  [bm25] index: {n} docs x {V} terms, {self.W.nnz/1e6:.1f}M nnz "
                  f"({time.time() - t0:.1f}s)", flush=True)

    # -- query vector reused across calls to avoid a V-sized alloc per query --
    def _qvec(self, toks: Sequence[str]):
        buf, vocab = self._buf, self.vocab
        touched = []
        for w in toks:
            j = vocab.get(w)
            if j is not None:
                if buf[j] == 0.0:
                    touched.append(j)
                buf[j] += 1.0
        return buf, touched

    def scores(self, query: str, pool: Optional[np.ndarray] = None) -> np.ndarray:
        buf, touched = self._qvec(_tok(query))
        try:
            M = self.W if pool is None else self.W[pool]
            s = M.dot(buf)
        finally:
            for j in touched:
                buf[j] = 0.0
        return s


_BM25_CACHE: Dict[Any, BM25Index] = {}


def _fingerprint(docs: Sequence[str]) -> Any:
    n = len(docs)
    probe = (docs[0], docs[n // 2], docs[-1]) if n else ("",)
    return (n, hash(probe))


def get_bm25_index(cand_texts: Sequence[str], verbose: bool = True) -> BM25Index:
    """Build-once / reuse-forever index, keyed by a cheap content fingerprint."""
    key = _fingerprint(cand_texts)
    idx = _BM25_CACHE.get(key)
    if idx is None:
        idx = BM25Index(cand_texts, verbose=verbose)
        if len(_BM25_CACHE) > 3:
            _BM25_CACHE.clear()
        _BM25_CACHE[key] = idx
    return idx


def bm25_ranks(query_texts: List[str], cand_texts: List[str],
               gold: np.ndarray, pool: np.ndarray | None = None,
               verbose: bool = True, heartbeat: int = 2000) -> np.ndarray:
    """Rank of gold[i] for query_texts[i], restricted to pool[i] if given.

    Ties are counted as half, matching common.ranks_from_banks.
    """
    gold = np.asarray(gold)
    nq = len(query_texts)
    assert len(gold) == nq, f"gold {len(gold)} != queries {nq}"
    if pool is not None:
        pool = np.asarray(pool)
        assert pool.shape[0] == nq, f"pool rows {pool.shape[0]} != queries {nq}"

    bm = get_bm25_index(cand_texts, verbose=verbose)
    ranks = np.empty(nq, dtype=np.float64)
    t0 = time.time()
    for i, q in enumerate(query_texts):
        cand = pool[i] if pool is not None else None
        sc = bm.scores(q, cand)
        if cand is None:
            gv = sc[gold[i]]
            n_pool = bm.n_docs
        else:
            g = np.flatnonzero(cand == gold[i])
            if g.size == 0:                       # gold not in pool: worst rank
                ranks[i] = len(cand)
                continue
            gv = sc[g[0]]
            n_pool = len(cand)
        ranks[i] = 1.0 + (sc > gv).sum() + 0.5 * max((sc == gv).sum() - 1, 0)
        if verbose and heartbeat and (i + 1) % heartbeat == 0:
            el = time.time() - t0
            print(f"    [bm25] {i+1}/{nq} queries  {el:.0f}s  "
                  f"eta {el/(i+1)*(nq-i-1):.0f}s  (pool={n_pool})", flush=True)
    return ranks


def strip_ngram_overlap(query_texts: List[str], cand_texts: List[str],
                        n: int = 5) -> List[str]:
    """no-overlap control: delete shared n-grams from the query side."""
    out = []
    for q, c in zip(query_texts, cand_texts):
        qt, ct = _tok(q), _tok(c)
        bad = {tuple(ct[i:i + n]) for i in range(max(len(ct) - n + 1, 0))}
        keep, i = [], 0
        while i < len(qt):
            if tuple(qt[i:i + n]) in bad:
                i += n
            else:
                keep.append(qt[i]); i += 1
        out.append(" ".join(keep))
    return out


# ================================================================== hard pools
@torch.no_grad()
def hard_pool(X: torch.Tensor, categories: List[str], K: int,
              query_aspect: int = 0, seed: int = 0,
              query_idx: Optional[np.ndarray] = None,
              chunk: int = 512, verbose: bool = True) -> np.ndarray:
    """(nq, K+1) pools: gold at column 0, then its K nearest WITHIN-CATEGORY
    neighbours in candidate space, in DESCENDING similarity order.

    Because the neighbours are sorted, pools for any K' < K are exactly
    `pools[:, :K'+1]` -- build once at K_max and slice. Chunked over queries so
    peak memory is chunk x |largest category|, not |category|^2.
    """
    t0 = time.time()
    Z = F.normalize(X[:, query_aspect, :].float(), dim=-1).to(device())
    n = Z.shape[0]
    cats = np.asarray(categories)
    if len(cats) != n:
        raise ValueError(f"categories {len(cats)} != n {n}")
    qidx = np.arange(n) if query_idx is None else np.asarray(query_idx)
    rng = np.random.default_rng(seed)

    pools = np.zeros((len(qidx), K + 1), dtype=np.int64)
    pools[:, 0] = qidx
    row_of = {int(g): r for r, g in enumerate(qidx)}
    qcats = cats[qidx]

    for c in np.unique(qcats):
        idx_all = np.flatnonzero(cats == c)          # sorted ascending
        q_in_c = qidx[qcats == c]
        k = int(min(K, len(idx_all) - 1))
        rows_all = np.fromiter((row_of[int(g)] for g in q_in_c),
                               dtype=np.int64, count=len(q_in_c))
        if k <= 0:                                   # singleton category
            extra = (q_in_c[:, None] + 1 + rng.integers(0, n - 1, size=(len(q_in_c), K))) % n
            pools[rows_all, 1:] = extra
            continue

        Csub = Z[torch.as_tensor(idx_all, device=Z.device)]
        for s in range(0, len(q_in_c), chunk):
            qg = q_in_c[s:s + chunk]
            rows = rows_all[s:s + chunk]
            Qs = Z[torch.as_tensor(qg, device=Z.device)]
            S = Qs @ Csub.T                                     # (b, |cat|)
            self_pos = np.searchsorted(idx_all, qg)
            S[torch.arange(len(qg), device=S.device),
              torch.as_tensor(self_pos, device=S.device)] = -2.0
            nn_ = S.topk(k, dim=1).indices.cpu().numpy()        # sorted desc
            pools[rows, 1:k + 1] = idx_all[nn_]
            del S, Qs
        if k < K:                                    # top up, never the gold
            extra = (q_in_c[:, None] + 1
                     + rng.integers(0, n - 1, size=(len(q_in_c), K - k))) % n
            pools[rows_all, k + 1:] = extra

    if verbose:
        print(f"  [pools] K={K} for {len(qidx)} queries over "
              f"{len(np.unique(qcats))} categories ({time.time() - t0:.1f}s)",
              flush=True)
    return pools


@torch.no_grad()
def pooled_ranks(Q: torch.Tensor, C: torch.Tensor, pools: np.ndarray,
                 query_idx: Optional[np.ndarray] = None,
                 max_elems: int = 48_000_000) -> torch.Tensor:
    """Ranks of pools[:, 0] within each pool. Chunk size adapts to K so that a
    K=1000 pool gathers at most `max_elems` floats at a time."""
    dev = device()
    Qn = F.normalize(Q.float(), dim=-1)
    if query_idx is not None:
        Qn = Qn[torch.as_tensor(np.asarray(query_idx), dtype=torch.long)]
    Qn = Qn.to(dev)
    Cn = F.normalize(C.float(), dim=-1).to(dev)
    P = torch.as_tensor(pools, dtype=torch.long, device=dev)
    if P.shape[0] != Qn.shape[0]:
        raise ValueError(f"pools {P.shape[0]} != queries {Qn.shape[0]}")

    d, K1 = Cn.shape[1], P.shape[1]
    chunk = int(max(1, min(1024, max_elems // max(K1 * d, 1))))
    out = torch.empty(Qn.shape[0], dtype=torch.float64)
    for i in range(0, Qn.shape[0], chunk):
        q, p = Qn[i:i + chunk], P[i:i + chunk]
        s = torch.einsum("bd,bkd->bk", q, Cn[p])
        g = s[:, :1]
        greater = (s > g).sum(1).double()
        ties = ((s == g).sum(1).double() - 1.0).clamp(min=0)
        out[i:i + chunk] = (1.0 + greater + 0.5 * ties).cpu()
        del s
    return out