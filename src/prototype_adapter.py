"""v10 scoring with warm-item k-means prototypes; immutable pre-test selection."""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans
from train_light_baseline import LightGCN
from protocol import load_data,sha256
from train_eval import seed_all
from formal_eval import make_episode,evaluate,GRID,write_json

def main(a):
    seed_all(a.seed);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    ck=torch.load(a.checkpoint,map_location='cuda',weights_only=False);data=load_data(a.dataset)
    assert ck['split']==data['meta'] and ck['args']['seed']==a.seed
    if a.model=='lightgcn':
        model=LightGCN(data['train'],ck['args']['dim'],ck['args']['layers']).cuda();model.load_state_dict(ck['model'])
        with torch.no_grad():items=model()[model.nu:].detach()
    else:
        # Exact even+odd official LightGCN output is the mean of all layers.
        model=LightGCN(data['train'],32,2).cuda()
        with torch.no_grad():
            model.embedding.weight.copy_(torch.cat([ck['model']['embedding_dict.user_emb'],ck['model']['embedding_dict.item_emb']]))
            items=model()[model.nu:].detach()
    warm_items=np.flatnonzero(np.asarray(data['train'].sum(0)).ravel()>0)
    vectors=items.cpu().numpy()
    km=KMeans(n_clusters=64,n_init=10,random_state=911,max_iter=100).fit(vectors[warm_items])
    proto=torch.as_tensor(km.cluster_centers_.T.copy(),device='cuda')
    np.savez_compressed(out/'prototypes.npz',centers=km.cluster_centers_,warm_items=warm_items)
    cache=dict(item_final=items,item_intent=proto)
    fit=make_episode(data,'fit',5,cache);tune=make_episode(data,'tune',5,cache,fit['scaler'])
    fixed=[dict(kind='fixed',lambda_=v) for v in GRID]
    fixed=[{'kind':'fixed','lambda':q['lambda_']} for q in fixed]
    fv,_=evaluate(fit,cache,fixed);short=np.argsort(-fv[:,:,0].mean(1))[:3]
    tv,_=evaluate(tune,cache,[fixed[i] for i in short]);chosen=fixed[short[int(tv[:,:,0].mean(1).argmax())]]
    specs=[dict(kind='candidate',lambda_=chosen['lambda'],alpha=alpha,coef=[aa,cc]) for alpha in [0.,.05,.1,.15,.2] for aa in [-1.,-.5,0.,.5,1.] for cc in [-1.,-.5,0.,.5,1.]]
    cv,_=evaluate(tune,cache,specs);selected=int(cv[:,:,0].mean(1).argmax());gate=specs[selected]
    np.savez_compressed(out/'validation.npz',fit_ids=fit['ids'],tune_ids=tune['ids'],fit_fixed=fv,tune_fixed=tv,tune_gate=cv)
    sources=['prototype_adapter.py','formal_eval.py','train_light_baseline.py','protocol.py','train_eval.py']
    for s in sources:shutil.copyfile(s,out/Path(s).name)
    lock=dict(model=a.model,dataset=a.dataset,seed=a.seed,checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_hash=sha256(a.checkpoint),split=data['meta'],fixed=chosen,gate=gate,selected=selected,scaler=fit['scaler'],prototype=dict(method='KMeans on raw warm-observed item embeddings',clusters=64,n_init=10,random_state=911,max_iter=100,temperature=1,hash=sha256(out/'prototypes.npz')),source_hashes={s:sha256(s) for s in sources},locked_at=time.time(),selection='v10: fixed fit top3 then tune; gate full 125-grid tune argmax',warning='adapted v10; original clamp [.05,.95] retained; development split')
    write_json(out/'lock.json',lock);del fit,tune,fv,tv,cv
    test=make_episode(data,'test',5,cache,lock['scaler']);names=['base','fixed','candidate_gate']
    ev=[{'kind':'base'},chosen,gate]
    vals,_=evaluate(test,cache,ev);arrays=dict(ids=test['ids']);metrics={}
    for name,v in zip(names,vals):
        metrics[name]={m:float(v[:,j].mean()) for j,m in enumerate(['ndcg','recall'])}
        for j,m in enumerate(['ndcg','recall']):arrays[name+'_'+m]=v[:,j]
    np.savez_compressed(out/'per_user.npz',**arrays)
    write_json(out/'result.json',dict(metrics=metrics,lock_hash=sha256(out/'lock.json'),test_users=len(test['ids']),version='v10-kmeans64 adapter; not native prototypes'))
    print(json.dumps(metrics),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['lightgcn','lightgcn_nt_ssm'],required=True)
    p.add_argument('--dataset',choices=['amazon'],required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,required=True)
    main(p.parse_args())
