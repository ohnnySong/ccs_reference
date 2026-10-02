#!/usr/bin/env python3
"""Regularized WTW-normalized opening-area specialist on cleaned features."""

from __future__ import annotations

import argparse,json,math,random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from ccs_model.metrics_v4 import area_metrics


class AreaHead(nn.Module):
    def __init__(self,dropout=.50):
        super().__init__(); self.network=nn.Sequential(nn.LayerNorm(769),nn.Linear(769,192),nn.GELU(),nn.Dropout(dropout),nn.Linear(192,1))
    def forward(self,global_feature,wtw):
        fraction=torch.sigmoid(self.network(torch.cat((global_feature,(wtw/11.3).unsqueeze(1)),1)).squeeze(1)); return fraction*wtw.square()


def load_data(feature_dir,manifest):
    chunks=[torch.load(p,map_location='cpu',weights_only=False) for p in sorted(feature_dir.glob('chunk_*.pt'))]; features=torch.cat([c['features'] for c in chunks])[:,:768]
    rows=[json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    if len(rows)!=len(features): raise ValueError('manifest and feature counts differ')
    area=torch.tensor([float(r['opening_area_mm2']) for r in rows]); wtw=torch.tensor([float(r['wtw_mm']) for r in rows]); train=torch.tensor([r['split']=='train' for r in rows]); val=torch.tensor([r['split']=='val' for r in rows]); return features,area,wtw,train,val


def evaluate(model,x,a,w):
    model.eval()
    with torch.no_grad(): pred=model(x,w)
    return area_metrics(a.tolist(),pred.tolist())


def main():
    p=argparse.ArgumentParser(); p.add_argument('--feature-dir',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--init-checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--epochs',type=int,default=250); p.add_argument('--patience',type=int,default=30); p.add_argument('--seed',type=int,default=20260927); args=p.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); x,a,w,train,val=load_data(args.feature_dir,args.manifest); xt,at,wt=x[train],a[train],w[train]; xv,av,wv=x[val],a[val],w[val]
    model=AreaHead(); source=torch.load(args.init_checkpoint,map_location='cpu',weights_only=False); state={k.removeprefix('area_fraction_head.'):v for k,v in source['state_dict'].items() if k.startswith('area_fraction_head.')}; model.network.load_state_dict(state,strict=True)
    for module in model.network:
        if isinstance(module,nn.Dropout): module.p=.50
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.10); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=2.5e-6); generator=torch.Generator().manual_seed(args.seed)
    args.output.mkdir(parents=True,exist_ok=False); history=[]; best=-math.inf; stale=0
    for epoch in range(1,args.epochs+1):
        model.train(); order=torch.randperm(len(xt),generator=generator); losses=[]
        for start in range(0,len(order),32):
            idx=order[start:start+32]; feature=xt[idx]+.01*torch.randn(xt[idx].shape,generator=generator); target=at[idx]; wtw=wt[idx]; optimizer.zero_grad(set_to_none=True); pred=model(feature,wtw); fraction_loss=F.smooth_l1_loss(pred/wtw.square(),target/wtw.square(),beta=.02); boundary=[]
            for threshold in (16.,20.): boundary.append(F.binary_cross_entropy_with_logits((pred-threshold)/2.,(target>threshold).float()))
            loss=3.*fraction_loss+.10*torch.stack(boundary).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); losses.append(float(loss.detach()))
        scheduler.step(); metrics=evaluate(model,xv,av,wv); score=.60*max(0.,1-metrics['mae_mm2']/10.)+.40*max(0.,1-metrics['rmse_mm2']/12.); record={'epoch':epoch,'train_loss':sum(losses)/len(losses),'selection_score':score,'metrics':metrics}; history.append(record)
        if score>best:
            best=score; stale=0; torch.save({'architecture':'ccs.area-feature-specialist.v1','head_state_dict':model.state_dict(),'epoch':epoch,'metrics':metrics,'wtw_formula':'area_mm2=predicted_fraction*wtw_mm^2'},args.output/'checkpoint_best.pt')
        else: stale+=1
        if epoch%10==0 or stale==0: print(json.dumps(record),flush=True)
        if stale>=args.patience: break
    (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n'); (args.output/'run_summary.json').write_text(json.dumps({'status':'complete','best_selection_score':best,'epochs_completed':len(history),'train_records':int(train.sum()),'validation_records':int(val.sum()),'test_accessed':False},indent=2)+'\n')
if __name__=='__main__': main()
