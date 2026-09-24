"""Shared cold-user split, immutable support/query, and full-catalog metrics."""
import hashlib
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(dataset, root=None, split_seed=20260915):
    if dataset != 'amazon':
        raise ValueError('The release package contains only the Amazon-Book protocol')
    default_root = Path(__file__).resolve().parents[1] / 'data'
    root = Path(root or default_root) / dataset
    # Only load the pinned official repository's trusted pickle files.
    matrices = []
    for name in ('train.pkl', 'test.pkl'):
        with (root / name).open('rb') as f:
            mat = pickle.load(f).tocsr().astype(np.float32)
        mat.sum_duplicates(); mat.sort_indices(); mat.data[:] = 1
        matrices.append(mat)
    tr, te = matrices
    assert tr.shape == te.shape and tr.multiply(te).nnz == 0
    users = np.flatnonzero((np.diff(tr.indptr) > 0) & (np.diff(te.indptr) > 0))
    rng = np.random.default_rng(split_seed)
    users = rng.permutation(users)
    nt, nc = int(len(users) * .2), int(len(users) * .1)
    test = np.sort(users[:nt]); cal = users[nt:nt+nc]
    fit = np.sort(cal[:nc//2]); tune = np.sort(cal[nc//2:])
    holdout = np.concatenate([test, fit, tune])
    warm_mask = np.diff(tr.indptr) > 0
    warm_mask[holdout] = False
    warm = np.flatnonzero(warm_mask)
    train_graph = tr[warm].tocsr()
    meta = dict(dataset=dataset, split_seed=split_seed, n_users=tr.shape[0],
                n_items=tr.shape[1], warm_users=len(warm), fit_users=len(fit),
                tune_users=len(tune), test_users=len(test),
                train_edges=train_graph.nnz,
                input_hashes={n: sha256(root/n) for n in ('train.pkl','test.pkl')})
    meta['split_hash'] = hashlib.sha256(b''.join(x.astype('<i8').tobytes()
        for x in (warm, fit, tune, test))).hexdigest()
    return dict(train=train_graph, support_source=tr, query=te, warm=warm,
                fit=fit, tune=tune, test=test, meta=meta)


def episodes(data, group, k, limit=None):
    """Nested random supports; never shuffle the query or condition on scores."""
    tr, te = data['support_source'], data['query']
    users = data[group]
    if limit:
        users = np.random.default_rng(1729).permutation(users)[:limit]
    ids, support, queries = [], [], []
    for u in users:
        available = tr.indices[tr.indptr[u]:tr.indptr[u+1]]
        q = te.indices[te.indptr[u]:te.indptr[u+1]]
        if len(available) < k or not len(q):
            continue
        rng = np.random.default_rng(np.random.SeedSequence([data['meta']['split_seed'], int(u), 73]))
        s = rng.permutation(available)[:k]
        assert not np.intersect1d(s, q).size
        ids.append(u); support.append(s); queries.append(q.copy())
    return np.array(ids), np.asarray(support, dtype=np.int64).reshape(-1, k), queries


def graph_config(train):
    nu, ni = train.shape
    z = train.tocoo()
    row = np.concatenate([z.row, z.col+nu])
    col = np.concatenate([z.col+nu, z.row])
    adj = sp.coo_matrix((np.ones(len(row),np.float32),(row,col)),shape=(nu+ni,nu+ni)).tocsr()
    z = adj.tocoo()
    return dict(n_users=nu, n_items=ni, plain_adj=adj,
                all_h_list=z.row.tolist(), all_t_list=z.col.tolist())


def rank_metrics(scores, support, queries, topk=20):
    """One implementation for aggregate, per-user and analysis records."""
    scores = scores.clone()
    support = torch.as_tensor(support, device=scores.device)
    scores.scatter_(1, support, -torch.inf)
    ranks = scores.topk(min(topk, scores.shape[1]), dim=1).indices.cpu().numpy()
    discount = 1 / np.log2(np.arange(ranks.shape[1]) + 2)
    recalls, ndcgs = [], []
    for rank, q in zip(ranks, queries):
        hit = np.isin(rank, q)
        recalls.append(float(hit.sum()/len(q)))
        ndcgs.append(float((hit*discount).sum()/discount[:min(len(q),len(discount))].sum()))
    return np.asarray(ndcgs), np.asarray(recalls)
