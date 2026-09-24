"""LightGCN-BPR on warm-only graph with shared support-mean cold adapter."""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from protocol import load_data,episodes,graph_config,rank_metrics,sha256
from train_eval import seed_all

class LightGCN(torch.nn.Module):
    def __init__(self,train,dim=32,layers=2):
        super().__init__(); self.nu,self.ni=train.shape; self.layers=layers
        self.embedding=torch.nn.Embedding(self.nu+self.ni,dim)
        torch.nn.init.normal_(self.embedding.weight,std=.1)
        cfg=graph_config(train); rows=torch.tensor(cfg['all_h_list']); cols=torch.tensor(cfg['all_t_list'])
        deg=torch.bincount(rows,minlength=self.nu+self.ni).float()
        val=(deg[rows]*deg[cols]).rsqrt()
        self.register_buffer('adj',torch.sparse_coo_tensor(torch.stack([rows,cols]),val,(len(deg),len(deg))).coalesce())
    def forward(self):
        x=self.embedding.weight; out=x
        for _ in range(self.layers): x=torch.sparse.mm(self.adj,x); out=out+x
        return out/(self.layers+1)

def sample_fast(train,n,rng):
    users=rng.integers(train.shape[0],size=n)
    start=train.indptr[users]; lengths=train.indptr[users+1]-start
    pos=train.indices[start+(rng.random(n)*lengths).astype(int)]
    neg=rng.integers(train.shape[1],size=n)
    bad=np.asarray(train[users,neg]).ravel()>0
    while bad.any():
        neg[bad]=rng.integers(train.shape[1],size=int(bad.sum()))
        bad=np.asarray(train[users,neg]).ravel()>0
    return users,pos,neg

@torch.no_grad()
def evaluate(items,episode):
    ids,support,query=episode; vals=[]
    for st in range(0,len(ids),128):
        s=support[st:st+128]; u=items[torch.as_tensor(s,device=items.device)].mean(1)
        n,r=rank_metrics(u@items.T,s,query[st:st+128]); vals.append(np.stack([n,r],1))
    return np.concatenate(vals)

def main(a):
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False); seed_all(a.seed)
    data=load_data(a.dataset); train=data['train']; tune=episodes(data,'tune',5)
    model=LightGCN(train,a.dim,a.layers).cuda(); opt=torch.optim.Adam(model.parameters(),lr=.001)
    rng=np.random.default_rng(a.seed); best=-1; best_epoch=0; history=[]
    for epoch in range(1,a.epochs+1):
        t=time.time(); us,ps,ns=sample_fast(train,a.batch*a.batches,rng); losses=[]
        for st in range(0,len(us),a.batch):
            u=torch.as_tensor(us[st:st+a.batch],device='cuda'); p=torch.as_tensor(ps[st:st+a.batch]+model.nu,device='cuda'); n=torch.as_tensor(ns[st:st+a.batch]+model.nu,device='cuda')
            emb=model(); ue,pe,ne=emb[u],emb[p],emb[n]
            reg=(model.embedding(u).square().sum()+model.embedding(p).square().sum()+model.embedding(n).square().sum())/(2*len(u))
            loss=F.softplus((ue*ne).sum(1)-(ue*pe).sum(1)).mean()+a.reg*reg
            assert torch.isfinite(loss); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        row=dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.time()-t)
        if epoch%a.eval_every==0 or epoch==a.epochs:
            val=evaluate(model()[model.nu:].detach(),tune).mean(0); row['tune']=val.tolist()
            if val[0]>best:
                best=float(val[0]);best_epoch=epoch
                torch.save(dict(model=model.state_dict(),args=vars(a),split=data['meta'],epoch=epoch),out/'best.pt')
        history.append(row)
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(row,flush=True)
        if epoch-best_epoch>=a.patience and best_epoch>0:break
    sources=['train_light_baseline.py','protocol.py','train_eval.py']
    for s in sources:shutil.copyfile(s,out/s)
    lock=dict(args=vars(a),split=data['meta'],best_epoch=best_epoch,tune_ndcg=best,checkpoint_hash=sha256(out/'best.pt'),source_hashes={s:sha256(s) for s in sources},selection='tune NDCG, k5; test evaluated only after lock',previously_explored_split=True)
    (out/'lock.json').write_text(json.dumps(lock,indent=2))
    ck=torch.load(out/'best.pt',weights_only=False);model.load_state_dict(ck['model'])
    items=model()[model.nu:].detach(); metrics={};arrays={}
    for k in [1,3,5,10]:
        ep=episodes(data,'test',k);v=evaluate(items,ep);metrics[str(k)]={m:float(v[:,j].mean()) for j,m in enumerate(['ndcg','recall'])}
        arrays[f'ids_k{k}']=ep[0]
        for j,m in enumerate(['ndcg','recall']):arrays[f'{m}_k{k}']=v[:,j]
    np.savez_compressed(out/'per_user.npz',**arrays)
    (out/'result.json').write_text(json.dumps(dict(metrics=metrics,lock_hash=sha256(out/'lock.json'),best_epoch=best_epoch),indent=2))
    print(metrics,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['dataset','out']:p.add_argument('--'+n,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--batch',type=int,default=10240);p.add_argument('--batches',type=int,default=40)
    p.add_argument('--dim',type=int,default=32);p.add_argument('--layers',type=int,default=2)
    p.add_argument('--reg',type=float,default=1e-4);p.add_argument('--eval_every',type=int,default=10);p.add_argument('--patience',type=int,default=50)
    main(p.parse_args())
