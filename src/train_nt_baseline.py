"""Official NT-SSM loss/encoder with project cold-user protocol adapter."""
import argparse,ast,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from train_light_baseline import evaluate,sample_fast
from train_eval import seed_all
from protocol import load_data,episodes,graph_config,sha256

UP=Path('upstream/NT-SSM-code')
SOURCE=UP/'model/graph/LightGCN_NT.py'

def official_components():
    """Execute original AST definitions, omitting framework imports only."""
    tree=ast.parse(SOURCE.read_text()); encoder=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LGCN_Encoder')
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LightGCN_NT')
    loss=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='ssm_loss')
    class Interface:
        @staticmethod
        def convert_sparse_mat_to_tensor(a):
            a=a.tocoo();return torch.sparse_coo_tensor(torch.tensor(np.stack([a.row,a.col])),torch.tensor(a.data,dtype=torch.float32),a.shape).coalesce()
    namespace=dict(torch=torch,nn=nn,F=F,TorchGraphInterface=Interface)
    exec(compile(ast.Module(body=[encoder,loss],type_ignores=[]),str(SOURCE),'exec'),namespace)
    return namespace['LGCN_Encoder'],namespace['ssm_loss']

def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);seed_all(a.seed)
    data=load_data(a.dataset);tr=data['train'];cfg=graph_config(tr)
    import scipy.sparse as sp
    adj=cfg['plain_adj']; degree=np.asarray(adj.sum(1)).ravel();inv=np.zeros_like(degree);inv[degree>0]=degree[degree>0]**-.5
    norm=sp.diags(inv)@adj@sp.diags(inv)
    Encoder,loss_fn=official_components();model=Encoder(SimpleNamespace(user_num=tr.shape[0],item_num=tr.shape[1],norm_adj=norm),32,2).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(a.seed);tune=episodes(data,'tune',5)
    best=-1;best_epoch=0
    for ep in range(1,a.epochs+1):
        t=time.time();u,p,n=sample_fast(tr,a.batch*a.batches,rng);losses=[]
        for st in range(0,len(u),a.batch):
            us=torch.as_tensor(u[st:st+a.batch],device='cuda');ps=torch.as_tensor(p[st:st+a.batch],device='cuda');ns=torch.as_tensor(n[st:st+a.batch],device='cuda')
            ue,uo,ie,io=model()
            loss=(loss_fn(None,ue[us],uo[us],ie[ps],io[ps],a.alpha_iu,a.alpha_ii,a.tau)+loss_fn(None,ie[ps],io[ps],ue[us],uo[us],a.alpha_uu,a.alpha_ui,a.tau))/2
            # Exact official l2_reg_loss and outer division by batch_size.
            em=model.embedding_dict
            loss+=a.reg*sum(torch.norm(x,p=2)/x.shape[0] for x in [em['user_emb'][us],em['item_emb'][ps],em['item_emb'][ns]])/len(us)
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        row=dict(epoch=ep,loss=float(np.mean(losses)),seconds=time.time()-t)
        if ep%10==0 or ep==a.epochs:
            with torch.no_grad(): _,_,ie,io=model();v=evaluate(ie+io,tune).mean(0)
            row['tune']=v.tolist()
            if v[0]>best:
                best=float(v[0]);best_epoch=ep;torch.save(dict(model=model.state_dict(),args=vars(a),epoch=ep,split=data['meta']),out/'best.pt')
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(row,flush=True)
        if ep-best_epoch>=50 and best_epoch>0:break
    sources=['train_nt_baseline.py','train_light_baseline.py','protocol.py','train_eval.py',str(SOURCE)]
    hashes={s:sha256(s) for s in sources}
    for s in sources:
        dest=out/'source'/s;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(s,dest)
    lock=dict(args=vars(a),split=data['meta'],best_epoch=best_epoch,tune_ndcg=best,checkpoint_hash=sha256(out/'best.pt'),source_hashes=hashes,upstream_commit=subprocess.check_output(['git','-C',str(UP),'rev-parse','HEAD'],text=True).strip(),adaptations=['warm-only graph','support-mean cold interface','uniform-user sampler and fixed samples/epoch shared with project','32 dimensions, 2 layers','tune-k5 selection instead of warm validation'],selection='tune NDCG before test',previously_explored_split=True)
    (out/'lock.json').write_text(json.dumps(lock,indent=2));model.load_state_dict(torch.load(out/'best.pt',weights_only=False)['model'])
    with torch.no_grad():_,_,ie,io=model();items=ie+io
    metrics={};arrays={}
    for k in [1,3,5,10]:
        episode=episodes(data,'test',k);v=evaluate(items,episode);metrics[str(k)]={m:float(v[:,j].mean()) for j,m in enumerate(['ndcg','recall'])};arrays[f'ids_k{k}']=episode[0]
        for j,m in enumerate(['ndcg','recall']):arrays[f'{m}_k{k}']=v[:,j]
    np.savez_compressed(out/'per_user.npz',**arrays)
    (out/'result.json').write_text(json.dumps(dict(metrics=metrics,lock_hash=sha256(out/'lock.json'),best_epoch=best_epoch),indent=2));print(metrics,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['dataset','out']:p.add_argument('--'+n,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--batch',type=int,default=2048);p.add_argument('--batches',type=int,default=200)
    p.add_argument('--reg',type=float,default=1e-4);p.add_argument('--tau',type=float,default=.2)
    for n,v in [('uu',1.2),('ii',.8),('ui',.8),('iu',1.)]:p.add_argument('--alpha_'+n,type=float,default=v)
    main(p.parse_args())
