"""Official DCCF/BIGCF training plus a common support-only cold-user adapter.

The adapter is deliberately separate from warm-start training: the official
model is trained on warm users only, and a cold user's representation consumes
only its support item embeddings. Fixed and adaptive fusion share these exact
two endpoint representations.
"""
import argparse, csv, json, random, time, math, os
from pathlib import Path
import numpy as np
import torch

from protocol import load_data, episodes, graph_config, rank_metrics
from models import build, arguments, layer_cache


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True


def sample(train, n, seed):
    nu, ni=train.shape; rows=[]; coo=train.tocoo()
    by=[[] for _ in range(nu)]
    for u,i in zip(coo.row,coo.col): by[int(u)].append(int(i))
    rng=np.random.default_rng(seed)
    us=rng.integers(0,nu,n)
    for u in us:
        p=by[int(u)][rng.integers(len(by[int(u)]))]
        q=int(rng.integers(ni))
        while q in by[int(u)]: q=int(rng.integers(ni))
        rows.append((int(u),p,q))
    return np.asarray(rows,np.int64),by


def train_model(name,data,epochs,batch,seed,log,n_batches=40,checkpoint=None):
    seed_all(seed); config=graph_config(data['train']); args=arguments(name,batch)
    model=build(name,config,args); opt=torch.optim.Adam(model.parameters(),lr=.001)
    history=[]
    if n_batches == 0:
        n_batches = math.ceil(data['train'].nnz / batch)
    for ep in range(epochs):
        pairs,by=sample(data['train'],batch*n_batches, seed*100000+ep); rng=np.random.default_rng(seed+ep)
        rng.shuffle(pairs); losses=[]; t=time.time(); model.train()
        for st in range(0,len(pairs),batch):
            b=pairs[st:st+batch]; opt.zero_grad(set_to_none=True)
            vals=model(b[:,0],b[:,1],b[:,2]); loss=sum(vals)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at epoch {ep+1}')
            loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        row=dict(epoch=ep+1,loss=float(np.mean(losses)),seconds=time.time()-t,
                 peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
        history.append(row); print(name,seed,row,flush=True)
        if log: log.write(json.dumps(row)+'\n'); log.flush()
        if checkpoint and ((ep+1)%5==0 or ep+1==epochs):
            state=dict(model=model.state_dict(),epoch=ep+1,optimizer=opt.state_dict(),
                       args=vars(args),name=name,split=data['meta'],seed=seed,
                       batches_per_epoch=n_batches,samples_per_epoch=batch*n_batches)
            torch.save(state,checkpoint)
    return model,history


@torch.no_grad()
def endpoints(model,name,data,group,k,limit=None,cache=None,feature_scaler=None):
    model.eval()
    ids,supp,queries=episodes(data,group,k,limit); cache=cache or layer_cache(model,name)
    ie=cache['item_final']; proto=cache['item_intent']
    S=torch.as_tensor(supp,device='cuda'); base=ie[S].mean(1)
    # Common cold adapter: item-prototype readout averaged over support.
    # This is an ADDED deterministic cold-user interface, not native BIGCF
    # inference nor its Gaussian mean. Native BIGCF noise has expectation 0.
    # All comparison arms use this exact same endpoint. Whether it helps at
    # all, especially on BIGCF, is an empirical feasibility question.
    ip=torch.softmax(ie@proto,dim=1)@proto.T
    # Evidence-aware support aggregation: uncertain support items contribute
    # less to the intent endpoint.  This changes the support-only estimate,
    # rather than merely applying a nearly constant post-hoc scalar gate.
    item_entropy=-(torch.softmax(ie@proto,dim=1)*(torch.softmax(ie@proto,dim=1)+1e-8).log()).sum(1)/np.log(proto.shape[1])
    # Sharpening is deliberately fixed before evaluation (not learned from
    # test); a larger value makes the endpoint trust only low-entropy support
    # items when the support set contains mixed evidence.
    support_conf=torch.exp(-40.0*item_entropy[S])
    support_conf=support_conf/(support_conf.sum(1,keepdim=True)+1e-8)
    intent=(ip[S]*support_conf.unsqueeze(2)).sum(1)
    p=torch.softmax(ie[S]@proto,dim=2).mean(1)
    ent=-(p*(p+1e-8).log()).sum(1)/np.log(proto.shape[1])
    # Support-only evidence; no user ID, query feature or test metric enters.
    pi=torch.softmax(ie[S]@proto,dim=2)
    mean_h=-(pi*(pi+1e-8).log()).sum(2).mean(1)/np.log(proto.shape[1])
    unit=torch.nn.functional.normalize(ie[S],dim=2)
    coherence=unit.mean(1).norm(dim=1)
    disagreement=ent-mean_h
    # Evidence v2: all statistics are support-only.  Per-group standardising
    # makes the gate sensitive to relative evidence quality rather than the
    # almost constant absolute entropy produced by a large prototype bank.
    conf_mean=support_conf.mean(1)
    raw_features=torch.stack([torch.full_like(ent,np.log1p(k)),ent,disagreement,coherence,conf_mean],1)
    features=raw_features.clone()
    scaler=[]
    for j in range(1, features.shape[1]):
        if feature_scaler is None:
            mu=features[:,j].mean(); sd=features[:,j].std(unbiased=False).clamp_min(1e-6)
        else:
            mu=torch.as_tensor(feature_scaler[j-1][0],device=features.device,dtype=features.dtype)
            sd=torch.as_tensor(feature_scaler[j-1][1],device=features.device,dtype=features.dtype).clamp_min(1e-6)
        scaler.append([float(mu),float(sd)])
        features[:,j]=(features[:,j]-mu)/sd
    return dict(ids=ids,support=supp,queries=queries,base=base,intent=intent,
                items=ie,features=features,raw_features=raw_features,feature_scaler=scaler,
                entropy=ent.cpu().numpy())


def _metric_from_embeddings(user_base, user_intent, item_emb, support, queries,
                            weight, batch_size=64):
    """Chunked full-catalog ranking; exact equivalent of rank_metrics."""
    nd, rc = [], []
    n = user_base.shape[0]
    w = weight if torch.is_tensor(weight) else None
    for st in range(0, n, batch_size):
        en = min(st + batch_size, n)
        b = user_base[st:en] @ item_emb.T
        i = user_intent[st:en] @ item_emb.T
        if w is None:
            ww = weight
        else:
            ww = w[st:en, None]
        scores = (1 - ww) * b + ww * i
        nn, rr = rank_metrics(scores, support[st:en], queries[st:en])
        nd.extend(nn.tolist()); rc.extend(rr.tolist())
    return np.asarray(nd), np.asarray(rc)


def _metric_support_evidence(user_base, support_emb, item_emb, support, queries,
                             weight, batch_size=32):
    """Chunked max-support similarity scorer, using support as evidence."""
    nd, rc = [], []
    n = user_base.shape[0]
    for st in range(0, n, batch_size):
        en = min(st + batch_size, n)
        base = user_base[st:en] @ item_emb.T
        agg=os.environ.get('EVIDENCE_AGG','max')
        if agg.startswith('cos'):
            sims = torch.einsum('bkd,md->bkm',
                torch.nn.functional.normalize(support_emb[st:en],dim=2),
                torch.nn.functional.normalize(item_emb,dim=1))
        else:
            sims = torch.einsum('bkd,md->bkm', support_emb[st:en], item_emb)
        if agg == 'mean':
            evidence=sims.mean(dim=1)
        elif agg == 'top2':
            evidence=sims.topk(min(2,sims.shape[1]),dim=1).values.mean(dim=1)
        else:
            evidence=sims.max(dim=1).values
        if agg.startswith('cos'):
            # Exclude observed support items before per-user score scaling.
            # They are masked from ranking and must not affect eligible-item
            # mean/variance.
            base=base.clone(); evidence=evidence.clone()
            sm=torch.as_tensor(support[st:en],device=base.device)
            base.scatter_(1,sm,float('nan'))
            evidence.scatter_(1,sm,float('nan'))
            bmu=torch.nanmean(base,1,keepdim=True); esmu=torch.nanmean(evidence,1,keepdim=True)
            bsd=torch.nanstd(base,1,keepdim=True).clamp_min(1e-6); esd=torch.nanstd(evidence,1,keepdim=True).clamp_min(1e-6)
            base=(base-bmu)/bsd; evidence=(evidence-esmu)/esd
            base=base.nan_to_num(0.0); evidence=evidence.nan_to_num(0.0)
        ww = weight if not torch.is_tensor(weight) else weight[st:en, None]
        scores = (1-ww)*base + ww*evidence
        nn, rr = rank_metrics(scores, support[st:en], queries[st:en])
        nd.extend(nn.tolist()); rc.extend(rr.tolist())
    return np.asarray(nd), np.asarray(rc)


@torch.no_grad()
def run_eval(model,name,data,group,k,limit=None,locked=None,allow_search=False):
    if group == 'test' and locked is None and not allow_search:
        raise ValueError('test evaluation requires --locked_config; test search requires explicit --allow_search')
    if group != 'fit' and locked is None and not allow_search:
        raise ValueError('non-fit search requires explicit --allow_search; use only for tune')
    feature_scaler=locked.get('feature_scaler') if locked else None
    ep=endpoints(model,name,data,group,k,limit,feature_scaler=feature_scaler)
    ids,supp,q,e=ep['ids'],ep['support'],ep['queries'],ep['entropy']
    # Do not materialize the full user-by-item score matrix. This is critical
    # for Amazon-Book (78k items) at full calibration/test scale.
    base_u, intent_u, item_e = ep['base'], ep['intent'], ep['items']
    support_e = item_e[torch.as_tensor(supp, device=item_e.device)]
    rows=[]
    fixed_grid=(np.unique(np.r_[np.linspace(0,1,21), np.arange(.025,1,.05), .99,.995,.999])) if locked is None else []
    for lam in fixed_grid:
        n,r=_metric_support_evidence(base_u,support_e,item_e,supp,q,float(lam))
        rows.append(dict(method='fixed',lambda_=float(lam),ndcg=float(n.mean()),recall=float(r.mean())))
    best=max(rows,key=lambda x:x['ndcg']) if locked is None else dict(lambda_=locked['fixed_lambda'])
    original_rows=[]
    for lam in fixed_grid if locked is None else []:
        n,r=_metric_from_embeddings(base_u,intent_u,item_e,supp,q,float(lam))
        original_rows.append(dict(method='original_fixed_intent',lambda_=float(lam),ndcg=float(n.mean()),recall=float(r.mean())))
    if locked is None:
        original_best=max(original_rows,key=lambda x:x['ndcg'])
    else:
        if 'original_fixed_lambda' not in locked:
            raise ValueError('lock missing original_fixed_lambda')
        original_best=dict(lambda_=locked['original_fixed_lambda'])
    # Evidence Gate family selected only from the support-derived feature
    # vector. Coefficients are kept tiny and the intercept is calibrated to
    # the fixed grid; this is a transparent reliability gate, not a second
    # recommender. The final choice must be validated on data['tune'].
    X=ep['features'].cpu().numpy()
    gate_rows=[]
    # Compact pre-registered v2 grid: includes the constant/count-only arms
    # while keeping development search tractable on the full catalog.
    if os.environ.get('FAST_SEARCH') == '1' and locked is None:
        gate_specs=[(0.0,0.0,0.0,0.0,0.0)]
    else:
        gate_specs=[(s,e,c,d,f) for s in ([0.0,0.5,1.0] if locked is None else [])
                    for e in ([-1.0,0.0,1.0] if locked is None else [])
                    for c in ([-1.0,0.0,1.0] if locked is None else [])
                    for d in ([0.0] if locked is None else [])
                    for f in ([-2.0,-1.0,0.0,1.0,2.0] if locked is None else [])]
    for slope,ent_coef,coh_coef,dis_coef,conf_coef in gate_specs:
      if locked is None:
        intercepts=[-2.0,-1.0,0.0,1.0,2.0]
      else:
        intercepts=[]
      for intercept in intercepts:
                z=intercept+slope*X[:,0]-ent_coef*X[:,1]+dis_coef*X[:,2]+coh_coef*X[:,3]+conf_coef*X[:,4]
                g=1/(1+np.exp(-z)); gt=torch.as_tensor(g,device=base_u.device,dtype=base_u.dtype)
                n,r=_metric_support_evidence(base_u,support_e,item_e,supp,q,gt)
                gate_rows.append(dict(method='gate',slope=slope,entropy_coef=ent_coef,intercept=intercept,
                                      coherence_coef=coh_coef,disagreement_coef=dis_coef,confidence_coef=conf_coef,
                                      ndcg=float(n.mean()),recall=float(r.mean()),mean_gate=float(g.mean()),
                                      gate_std=float(g.std())))
    best_gate=max(gate_rows,key=lambda x:x['ndcg']) if locked is None else locked['gate']
    z=best_gate['intercept']+best_gate['slope']*X[:,0]-best_gate['entropy_coef']*X[:,1]+best_gate.get('disagreement_coef',0.0)*X[:,2]+best_gate.get('coherence_coef',0.0)*X[:,3]+best_gate.get('confidence_coef',0.0)*X[:,4]
    g=1/(1+np.exp(-z)); gt=torch.as_tensor(g,device=base_u.device,dtype=base_u.dtype)
    # Per-user oracle headroom is diagnostic only; never used to select test.
    fn,fr=_metric_support_evidence(base_u,support_e,item_e,supp,q,float(best['lambda_']))
    gn,gr=_metric_support_evidence(base_u,support_e,item_e,supp,q,gt)
    bn,br=_metric_from_embeddings(base_u,intent_u,item_e,supp,q,0.0)
    ofn,ofr=_metric_from_embeddings(base_u,intent_u,item_e,supp,q,float(original_best['lambda_']))
    return dict(ids=ids,support=supp,queries=q,base_ndcg=bn,
      base_recall=br,
      intent_ndcg=_metric_from_embeddings(base_u,intent_u,item_e,supp,q,1.0)[0],
      original_fixed_ndcg=ofn,original_fixed_recall=ofr,
      fixed_ndcg=fn,fixed_recall=fr,gate_ndcg=gn,gate_recall=gr,entropy=e,grid=rows,best_fixed=best,
      original_grid=original_rows,original_best=original_best,
      endpoint=ep,best_gate=best_gate,gate_rows=gate_rows)


def main(a):
    seed_all(a.seed)
    os.environ['EVIDENCE_AGG']=a.evidence_agg
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    data=load_data(a.dataset,split_seed=a.split_seed)
    (out/f'{a.name}_split.json').write_text(json.dumps(data['meta'],indent=2))
    config=graph_config(data['train']); argsm=arguments(a.model,a.batch)
    if a.load_checkpoint:
        model=build(a.model,config,argsm)
        ck=torch.load(a.load_checkpoint,map_location='cuda',weights_only=False)
        if ck.get('split') != data['meta'] or ck.get('name') != a.model or ck.get('seed') != a.seed:
            raise ValueError('checkpoint split/model/seed mismatch')
        state=ck['model'] if 'model' in ck else ck; model.load_state_dict(state)
        h=[]
    else:
        with (out/f'{a.name}_train.jsonl').open('w') as log:
            model,h=train_model(a.model,data,a.epochs,a.batch,a.seed,log,a.n_batches,out/f'{a.name}_checkpoint.pt')
    # Save checkpoint and development outputs; test is intentionally not used
    # here unless --eval_group=test is explicitly requested after calibration.
    (out/f'{a.name}_config.json').write_text(json.dumps(vars(a),indent=2))
    if a.train_only:
        return
    locked=json.loads(Path(a.locked_config).read_text()) if a.locked_config else None
    if locked:
        for key,val in [('version',2),('model',a.model),('dataset',a.dataset),('k',a.k),('split_seed',a.split_seed),('evidence_agg',a.evidence_agg)]:
            if locked.get(key) != val:
                raise ValueError(f'lock provenance mismatch for {key}: {locked.get(key)!r} != {val!r}')
    ev=run_eval(model,a.model,data,a.eval_group,a.k,a.limit,locked,a.allow_search)
    (out/f'{a.name}_{a.eval_group}_grid.json').write_text(json.dumps(ev['grid'],indent=2))
    np.savez_compressed(out/f'{a.name}_{a.eval_group}_k{a.k}.npz',
      ids=ev['ids'],support=ev['support'],entropy=ev['entropy'],
      base_ndcg=ev['base_ndcg'],base_recall=ev['base_recall'],
      intent_ndcg=ev['intent_ndcg'],fixed_ndcg=ev['fixed_ndcg'],
      original_fixed_ndcg=ev['original_fixed_ndcg'],original_fixed_recall=ev['original_fixed_recall'],
      fixed_recall=ev['fixed_recall'],gate_ndcg=ev['gate_ndcg'],gate_recall=ev['gate_recall'])
    print(json.dumps({'name':a.name,'dataset':a.dataset,'model':a.model,
      'group':a.eval_group,'k':a.k,'best_fixed':ev['best_fixed'],
      'base_ndcg':float(ev['base_ndcg'].mean()),
      'fixed_ndcg':float(ev['fixed_ndcg'].mean()),
      'original_fixed_ndcg':float(ev['original_fixed_ndcg'].mean()),
      'gate_ndcg':float(ev['gate_ndcg'].mean()),
      'fixed_recall':float(ev['fixed_recall'].mean()),
      'original_fixed_recall':float(ev['original_fixed_recall'].mean()),
      'gate_recall':float(ev['gate_recall'].mean()),'best_gate':ev['best_gate']},indent=2),flush=True)
    if a.save_locked:
        if a.eval_group == 'test':
            raise ValueError('refuse to save a test-selected configuration')
        cfg={'version':2,'model':a.model,'dataset':a.dataset,'k':a.k,
             'split_seed':a.split_seed,'evidence_agg':a.evidence_agg,
             'fixed_lambda':float(ev['best_fixed']['lambda_']),
             'original_fixed_lambda':float(ev['original_best']['lambda_']),
             'feature_scaler':ev['endpoint']['feature_scaler'],
             'gate':{k:v for k,v in ev['best_gate'].items()
                     if k in ('method','slope','entropy_coef','intercept','coherence_coef','disagreement_coef','confidence_coef')}}
        Path(a.save_locked).write_text(json.dumps(cfg,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['dccf','bigcf'],required=True); p.add_argument('--dataset',choices=['amazon'],required=True); p.add_argument('--name',required=True); p.add_argument('--out',default='runs'); p.add_argument('--epochs',type=int,default=1); p.add_argument('--batch',type=int,default=10240); p.add_argument('--n_batches',type=int,default=40); p.add_argument('--seed',type=int,default=0); p.add_argument('--split_seed',type=int,default=20260915); p.add_argument('--eval_group',choices=['fit','tune','test'],default='fit'); p.add_argument('--k',type=int,default=5); p.add_argument('--limit',type=int,default=0); p.add_argument('--load_checkpoint',default=''); p.add_argument('--locked_config',default=''); p.add_argument('--save_locked',default=''); p.add_argument('--allow_search',action='store_true'); p.add_argument('--evidence_agg',choices=['max','mean','top2','cosmax'],default='max'); p.add_argument('--train_only',action='store_true'); main(p.parse_args())
