"""Load pinned official models. Only replace adjacency normalization kernels."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def official_class(name):
    relative = 'DCCF/model.py' if name == 'dccf' else 'BIGCF/BIGCF.py'
    path = Path(__file__).parent/'upstream'/relative
    spec = importlib.util.spec_from_file_location('official_'+name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return getattr(module, name.upper())


def arguments(name, batch_size=1024):
    return SimpleNamespace(embed_size=32, n_layers=2,
        n_intents=128 if name=='dccf' else 64,
        temp=1.0 if name=='dccf' else .2, batch_size=batch_size,
        emb_reg=2.5e-5 if name=='dccf' else 1e-5,
        cen_reg=5e-3 if name=='dccf' else 1e-5,
        ssl_reg=.1 if name=='dccf' else .4)


def build(name, config, args):
    parent = official_class(name)
    class SafeAdjacency(parent):
        def _cal_sparse_adj(self):
            # D^(-1/2) A D^(-1/2) on existing edges only; no inf at
            # isolated catalog items and no expensive sparse-sparse product.
            row, col = self.all_h_list, self.all_t_list
            degree = torch.bincount(row, minlength=self.n_users+self.n_items)
            values = (degree[row].float()*degree[col].float()).rsqrt()
            return torch.stack((row,col)), values
    return SafeAdjacency(config,args).cuda()


@torch.no_grad()
def layer_cache(model, name):
    """Replay official inference equations and expose deterministic layers.

    BIGCF's Gaussian draw is retained separately; its intent is a noise scale,
    not a replacement deterministic user mean.
    """
    import torch_sparse
    nu, ni = model.n_users, model.n_items
    x = torch.cat((model.user_embedding.weight,model.item_embedding.weight))
    layers=[x]; detail=[]
    for _ in range(model.n_layers):
        g = torch_sparse.spmm(model.G_indices,model.G_values,len(x),len(x),x)
        if name=='dccf':
            up = (x[:nu]@model.user_intent).softmax(1)
            ip = (x[nu:]@model.item_intent).softmax(1)
            intent = torch.cat((up@model.user_intent.T, ip@model.item_intent.T))
            idx,val=model._adaptive_mask(g[model.all_h_list],g[model.all_t_list])
            ga = torch_sparse.spmm(idx,val,len(x),len(x),x)
            idx,val=model._adaptive_mask(intent[model.all_h_list],intent[model.all_t_list])
            ia = torch_sparse.spmm(idx,val,len(x),len(x),x)
            detail.append(dict(gnn=g[nu:],intent=intent[nu:]))
            x = x+g+intent+ga+ia
        else:
            x=g
        layers.append(x)
    total=torch.stack(layers).sum(0)
    return dict(items=[z[nu:].clone() for z in layers], detail=detail,
                item_final=total[nu:].clone(), user_final=total[:nu].clone(),
                user_intent=model.user_intent.clone(),item_intent=model.item_intent.clone())
