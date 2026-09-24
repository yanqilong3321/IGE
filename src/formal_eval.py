"""Version 3 evaluation: fit -> tune -> immutable lock -> test.

Cold-user adapters on official warm-trained encoders, NOT native published
cold-user algorithms. The historical split is a reused development benchmark.
No candidate generation, normalization fitting or selection consumes test labels.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from models import build, layer_cache
from protocol import load_data, episodes, graph_config, sha256
from train_eval import seed_all

VERSION = 4
FEATURES = ['prototype_entropy', 'prototype_disagreement', 'support_coherence',
            'candidate_intent_alignment', 'global_local_agreement']
GRID = sorted(set(np.linspace(0, 1, 41).tolist()+[.99,.995,.999]))


def write_json(path, obj):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def masked_zscore(scores, support):
    """Population statistics over eligible items only (support is excluded)."""
    masked=scores.clone().scatter(1,support,0)
    n=scores.shape[1]-support.shape[1]
    mu=masked.sum(1,keepdim=True)/n
    dev=(scores-mu).scatter(1,support,0)
    sd=(dev.square().sum(1,keepdim=True)/n).sqrt().clamp_min(1e-6)
    return ((scores-mu)/sd).scatter(1,support,0)


def fast_metrics(scores, support, relevance, counts):
    ranks=scores.scatter(1,support,-torch.inf).topk(min(20,scores.shape[1]),dim=1).indices
    hit=relevance.gather(1,ranks).float()
    discount=1/torch.log2(torch.arange(hit.shape[1],device=scores.device)+2.)
    ideal=discount.cumsum(0)[counts.clamp(max=hit.shape[1])-1]
    return torch.stack(((hit*discount).sum(1)/ideal,hit.sum(1)/counts),dim=1)


def scale_features(raw, scaler=None):
    if scaler is None:
        scaler=dict(mean=raw.mean(0).tolist(), std=raw.std(0).clip(1e-6).tolist())
    return (raw-np.asarray(scaler['mean']))/np.asarray(scaler['std']),scaler


def weights(spec, features):
    if spec['kind'] != 'adaptive':
        return np.full(len(features), spec.get('lambda',0), dtype=np.float32)
    z=spec['intercept']+features@np.asarray(spec['coef'])
    return (1/(1+np.exp(-np.clip(z,-30,30)))).astype(np.float32)


@torch.no_grad()
def make_episode(data, group, k, cache, scaler=None, limit=0):
    ids,support,queries=episodes(data,group,k,limit)
    ie,proto=cache['item_final'],cache['item_intent']
    s=torch.as_tensor(support,device=ie.device)
    se=ie[s]
    probs=(se@proto).softmax(2)
    p=probs.mean(1)
    entropy=-(p*p.clamp_min(1e-30).log()).sum(1)/math.log(proto.shape[1])
    indiv=-(probs*probs.clamp_min(1e-30).log()).sum(2).mean(1)/math.log(proto.shape[1])
    coherence=F.normalize(se,dim=2).mean(1).norm(dim=1)
    # Candidate-level intent alignment: mean cosine similarity between the
    # support's latent intent distribution and eligible item intent vectors.
    # Global/local agreement: Pearson agreement between the two ranking scores.
    item_probs=(cache['item_final']@cache['item_intent']).softmax(1)
    item_intent=F.normalize(item_probs,dim=1)
    pnorm=F.normalize(p,dim=1)
    align=[]; agree=[]
    unit=F.normalize(ie,dim=1)
    for start in range(0,len(ids),256):
        end=min(len(ids),start+256); ss=s[start:end]
        b=se[start:end].mean(1)@ie.T
        r=torch.einsum('bkd,md->bkm',F.normalize(se[start:end],dim=2),unit).amax(1)
        # Exclude observed support items from summaries, matching ranking.
        mask=torch.zeros_like(b,dtype=torch.bool).scatter(1,ss,True)
        valid=~mask
        a=(pnorm[start:end]@item_intent.T).masked_fill(mask,0)
        align.append((a.sum(1)/valid.sum(1)).cpu())
        bz=masked_zscore(b,ss); rz=masked_zscore(r,ss)
        bm=bz.masked_fill(mask,0); rm=rz.masked_fill(mask,0)
        n=valid.sum(1).float(); bm=bm-bm.sum(1,keepdim=True)/n[:,None]; rm=rm-rm.sum(1,keepdim=True)/n[:,None]
        agree.append(((bm*rm).sum(1)/(bm.square().sum(1).sqrt()*rm.square().sum(1).sqrt()).clamp_min(1e-6)).cpu())
    raw=torch.stack([entropy.cpu(),(entropy-indiv).cpu(),coherence.cpu(),torch.cat(align),torch.cat(agree)],1).numpy()
    x,scaler=scale_features(raw,scaler)
    # Restore the ORIGINAL unweighted prototype-mean cold adapter explicitly.
    intent=(probs@proto.T).mean(1)
    return dict(ids=ids,support=s,queries=queries,base=se.mean(1),intent=intent,
                support_prob=p,
                support_emb=se,features=x,raw_features=raw,scaler=scaler)


@torch.no_grad()
def evaluate(ep, cache, specs, batch=64):
    n=len(ep['ids']); ie=cache['item_final']; unit=F.normalize(ie,dim=1)
    all_weights=np.stack([weights(s,ep['features']) for s in specs])
    values=np.empty((len(specs),n,2),np.float32)
    for start in range(0,n,batch):
        end=min(n,start+batch); supp=ep['support'][start:end]
        base=ep['base'][start:end]@ie.T
        intent=ep['intent'][start:end]@ie.T
        evidence=torch.einsum('bkd,md->bkm',F.normalize(ep['support_emb'][start:end],dim=2),unit).amax(1)
        zb=masked_zscore(base,supp); ze=masked_zscore(evidence,supp)
        candidate_alignment=(F.normalize(ep['support_prob'][start:end],dim=1) @
                             F.normalize((cache['item_final']@cache['item_intent']).softmax(1),dim=1).T)
        candidate_alignment=masked_zscore(candidate_alignment,supp)
        candidate_agreement=1-(zb-ze).abs()/2
        relevance=torch.zeros_like(base,dtype=torch.bool)
        q=ep['queries'][start:end]
        counts=torch.as_tensor([len(v) for v in q],device=ie.device)
        rows=np.repeat(np.arange(end-start),[len(v) for v in q])
        relevance[torch.as_tensor(rows,device=ie.device),torch.as_tensor(np.concatenate(q),device=ie.device)]=True
        chunk=[]
        for j,spec in enumerate(specs):
            w=torch.as_tensor(all_weights[j,start:end,None],device=ie.device)
            if spec['kind']=='base':
                score=base
            elif spec['kind']=='original':
                score=(1-w)*base+w*intent
            elif spec['kind']=='candidate':
                advantage=ze-zb
                z=spec['coef'][0]*candidate_alignment+spec['coef'][1]*advantage
                w=(spec['lambda_']+spec['alpha']*torch.tanh(z)).clamp(0.05,0.95)
                score=(1-w)*zb+w*ze
            else:
                score=(1-w)*zb+w*ze
            chunk.append(fast_metrics(score,supp,relevance,counts))
        values[:,start:end]=torch.stack(chunk).cpu().numpy()
    return values,all_weights


def adaptive_candidates(center, active=None):
    """Predeclared small gate grid; exact constants remain available separately."""
    intercept=math.log(np.clip(center,.025,.975)/(1-np.clip(center,.025,.975)))
    directions=[]
    if active is None: active=range(len(FEATURES))
    for j in active:
        for strength in [-1.,-.5,-.25,.25,.5,1.]:
            coef=[0.]*len(FEATURES); coef[j]=strength; directions.append(coef)
    directions += [[a,b,c,d,e] for a in [-.5,.5] for b in [-.5,.5] for c in [-.5,.5] for d in [-.5,.5] for e in [-.5,.5]]
    return [dict(kind='adaptive',intercept=intercept+shift,coef=c)
            for shift in [-.5,0.,.5] for c in directions]


def top_indices(values, n=3):
    return np.argsort(-values[:,:,0].mean(1),kind='stable')[:n].tolist()


def validate_lock(lock, provenance):
    if lock['provenance'] != provenance:
        raise ValueError('Lock/checkpoint/code/split provenance mismatch')


def mean_rows(specs, vals):
    return [dict(spec=s,ndcg=float(v[:,0].mean()),recall=float(v[:,1].mean()))
            for s,v in zip(specs,vals)]


def main(a):
    seed_all(a.seed); started=time.time()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    result=out/'result.json'
    if result.exists():
        raise FileExistsError('Preserve completed result: '+str(result))
    ck=torch.load(a.checkpoint,map_location='cuda',weights_only=False)
    data=load_data(a.dataset,split_seed=20260915)
    if ck['split']!=data['meta'] or ck['name']!=a.model or ck['seed']!=a.seed:
        raise ValueError('Checkpoint identity/split mismatch')
    model=build(a.model,graph_config(data['train']),SimpleNamespace(**ck['args']))
    model.load_state_dict(ck['model']); model.eval()
    cache=layer_cache(model,a.model)
    provenance=dict(version=VERSION,model=a.model,dataset=a.dataset,seed=a.seed,k=a.k,
        split=data['meta'],checkpoint=str(Path(a.checkpoint).resolve()),
        checkpoint_hash=sha256(Path(a.checkpoint)),epoch=ck['epoch'],
        batches_per_epoch=ck['batches_per_epoch'],evidence_agg='cosmax',limit=a.limit,
        runtime=dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda),
        code_hashes={p:sha256(Path(__file__).parent/p) for p in ['formal_eval.py','models.py','protocol.py',
            'train_eval.py','upstream/DCCF/model.py','upstream/BIGCF/BIGCF.py']})
    snapshot=out/'source'; snapshot.mkdir(exist_ok=True)
    for source,digest in provenance['code_hashes'].items():
        target=snapshot/source
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha256(target)!=digest:
            raise ValueError('Source snapshot mismatch; use a new output directory')
        if not target.exists(): shutil.copyfile(Path(__file__).parent/source,target)
    lockfile=out/'lock.json'
    if lockfile.exists():
        lock=json.loads(lockfile.read_text()); validate_lock(lock,provenance)
        print('Reusing immutable validation lock',flush=True)
    else:
        fit=make_episode(data,'fit',a.k,cache,limit=a.limit)
        fixed=[{'kind':'fixed','lambda':v} for v in GRID]
        original=[{'kind':'original','lambda':v} for v in GRID]
        print('Fit fixed grids',len(fit['ids']),flush=True)
        vals,_=evaluate(fit,cache,fixed+original,a.batch)
        fv,ov=vals[:len(fixed)],vals[len(fixed):]
        fi,oi=top_indices(fv),top_indices(ov)
        adaptive=adaptive_candidates(fixed[fi[0]]['lambda'])
        print('Fit adaptive grid',len(adaptive),flush=True)
        av,_=evaluate(fit,cache,adaptive,a.batch)
        ai=top_indices(av)
        shortlist=[fixed[j] for j in fi]+[original[j] for j in oi]+[adaptive[j] for j in ai]
        tune=make_episode(data,'tune',a.k,cache,fit['scaler'],a.limit)
        tv,_=evaluate(tune,cache,shortlist,a.batch)
        fbest=int(np.argmax(tv[:3,:,0].mean(1)))
        obest=3+int(np.argmax(tv[3:6,:,0].mean(1)))
        abest=6+int(np.argmax(tv[6:,:,0].mean(1)))
        # EXACT fixed fallback, including lambda=0/1, never approximated by sigmoid.
        chosen=abest if tv[abest,:,0].mean()>tv[fbest,:,0].mean() else fbest
        lock=dict(provenance=provenance,feature_names=FEATURES,scaler=fit['scaler'],
            methods=dict(base={'kind':'base'},original=shortlist[obest],
                         fixed=shortlist[fbest],gate=shortlist[chosen]),
            fit_users=len(fit['ids']),tune_users=len(tune['ids']),
            fit_fixed=mean_rows(fixed,fv),fit_original=mean_rows(original,ov),
            fit_adaptive=mean_rows(adaptive,av),tune=mean_rows(shortlist,tv),
            selection='Fit top 3 per family; tune selects fixed/original; gate competes with exact tuned fixed',
            test_previously_explored=True,locked_at=time.time())
        write_json(lockfile,lock)
        print('LOCKED',lock['methods'],flush=True)
        del fit,tune,vals,fv,ov,av,tv
    validate_lock(lock,provenance)
    # Candidate-conditioned gate: a separate, small fit/tune selection.  This
    # is deliberately not folded into the user-level lock so the v4 baseline
    # remains exactly reproducible while the structural variant is auditable.
    fit_c=make_episode(data,'fit',a.k,cache,limit=a.limit)
    tune_c=make_episode(data,'tune',a.k,cache,fit_c['scaler'],a.limit)
    center=lock['methods']['fixed'].get('lambda',.5)
    candidate_specs=[]
    for alpha in [0.,.05,.1,.15,.2]:
        for aa in [-1.,-.5,0.,.5,1.]:
            for cc in [-1.,-.5,0.,.5,1.]:
                candidate_specs.append(dict(kind='candidate',lambda_=center,alpha=alpha,coef=[aa,cc]))
    fit_cv,_=evaluate(fit_c,cache,candidate_specs,a.batch)
    fi=int(np.argmax(fit_cv[:,:,0].mean(1)))
    tune_cv,_=evaluate(tune_c,cache,candidate_specs,a.batch)
    ti=int(np.argmax(tune_cv[:,:,0].mean(1)))
    candidate_gate=candidate_specs[ti]
    candidate_gate_fit=float(fit_cv[fi,:,0].mean()); candidate_gate_tune=float(tune_cv[ti,:,0].mean())
    del fit_c,tune_c,fit_cv,tune_cv
    test=make_episode(data,'test',a.k,cache,lock['scaler'],a.limit)
    names=list(lock['methods']); specs=list(lock['methods'].values())
    # The release evaluator reports only the main comparison arms. Ablations
    # and diagnostics are intentionally kept out of this GitHub package.
    eval_names = ['base', 'fixed_fusion', 'evidence_gate']
    eval_specs = [
        {'kind': 'base'},
        lock['methods']['fixed'],
        candidate_gate,
    ]
    vals,gates=evaluate(test,cache,eval_specs,a.batch)
    arrays=dict(ids=test['ids'],support=test['support'].cpu().numpy(),features=test['raw_features'])
    metrics={}
    for name,v,w in zip(eval_names,vals,gates):
        metrics[name]=dict(ndcg=float(v[:,0].mean()),recall=float(v[:,1].mean()),
                           weight_mean=float(w.mean()),weight_std=float(w.std()))
        arrays[name+'_ndcg']=v[:,0]; arrays[name+'_recall']=v[:,1]; arrays[name+'_weight']=w
    np.savez_compressed(out/'per_user.npz',**arrays)
    diff=arrays['evidence_gate_ndcg']-arrays['fixed_fusion_ndcg']
    report=dict(provenance=provenance,metrics=metrics,test_users=len(test['ids']),
        gate_is_adaptive=lock['methods']['gate']['kind']=='adaptive',
        gate_vs_fixed_relative_pct=100*(metrics['evidence_gate']['ndcg']/metrics['fixed_fusion']['ndcg']-1),
        candidate_gate=candidate_gate,candidate_gate_fit_ndcg=candidate_gate_fit,
        candidate_gate_tune_ndcg=candidate_gate_tune,
        paired_difference_mean=float(diff.mean()),paired_user_se=float(diff.std(ddof=1)/np.sqrt(len(diff))),
        seconds=time.time()-started,warning='Reused development benchmark; not an independent untouched test.')
    write_json(result,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,choices=['dccf','bigcf'])
    p.add_argument('--dataset',required=True,choices=['amazon'])
    p.add_argument('--checkpoint',required=True); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--k',type=int,required=True); p.add_argument('--out',required=True)
    p.add_argument('--batch',type=int,default=64); p.add_argument('--limit',type=int,default=0)
    main(p.parse_args())
