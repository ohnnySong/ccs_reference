#!/usr/bin/env python3
"""Train the regularized opacity head from cached frozen image features."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from ccs_model.metrics_v4 import multiclass_metrics
from ccs_model.train_document_v1 import OPACITY_NAMES


class Head(nn.Module):
    def __init__(self,dropout=0.55,hidden=128):
        super().__init__(); self.network=nn.Sequential(nn.LayerNorm(2304),nn.Dropout(dropout),nn.Linear(2304,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,4))
    def forward(self,x): return self.network(x)


def metrics(model,x,y):
    model.eval()
    with torch.no_grad(): pred=model(x).argmax(1).tolist()
    return multiclass_metrics(y.tolist(),pred,OPACITY_NAMES)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--feature-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--epochs',type=int,default=200); p.add_argument('--patience',type=int,default=25); p.add_argument('--seed',type=int,default=20260923); args=p.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    chunks=[torch.load(path,map_location='cpu',weights_only=False) for path in sorted(args.feature_dir.glob('chunk_*.pt'))]
    x=torch.cat([c['features'] for c in chunks]); y=torch.cat([c['labels'] for c in chunks]); splits=sum([c['splits'] for c in chunks],[])
    train=torch.tensor([s=='train' for s in splits]); val=torch.tensor([s=='val' for s in splits])
    x_train=x[train]; y_train=y[train]; x_val=x[val]; y_val=y[val]
    mean=x_train.mean(0); std=x_train.std(0).clamp_min(1e-5); x_train=(x_train-mean)/std; x_val=(x_val-mean)/std
    counts=torch.bincount(y_train,minlength=4).float(); weights=counts.pow(-0.35); weights/=weights.mean()
    model=Head(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.10); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=5e-6)
    args.output.mkdir(parents=True,exist_ok=False); history=[]; best=-math.inf; stale=0
    generator=torch.Generator().manual_seed(args.seed)
    for epoch in range(1,args.epochs+1):
        model.train(); order=torch.randperm(len(x_train),generator=generator); losses=[]
        for start in range(0,len(order),32):
            idx=order[start:start+32]; feature=x_train[idx]; target=y_train[idx]
            feature=feature+0.025*torch.randn(feature.shape,generator=generator)
            optimizer.zero_grad(set_to_none=True)
            if random.random()<0.5 and len(idx)>1:
                perm=torch.randperm(len(idx),generator=generator); lam=random.uniform(.65,.85); mixed=lam*feature+(1-lam)*feature[perm]; logits=model(mixed); one=F.one_hot(target,4).float(); soft=lam*one+(1-lam)*one[perm]; sample_weight=(soft*weights).sum(1); loss=(-(soft*F.log_softmax(logits,1)).sum(1)*sample_weight).mean()
            else:
                logits=model(feature); ce=F.cross_entropy(logits,target,weight=weights,label_smoothing=.08); levels=torch.arange(4,dtype=logits.dtype); expected=(logits.softmax(1)*levels).sum(1); loss=ce+.20*F.smooth_l1_loss(expected,target.float(),beta=.75)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach()))
        scheduler.step(); m=metrics(model,x_val,y_val); score=.7*m['macro_f1']+.3*m['balanced_accuracy']; record={'epoch':epoch,'train_loss':sum(losses)/len(losses),'selection_score':score,'metrics':m}; history.append(record)
        if score>best:
            best=score; stale=0; torch.save({'architecture':'ccs.opacity-feature-specialist.v2','head_state_dict':model.state_dict(),'feature_mean':mean,'feature_std':std,'epoch':epoch,'metrics':m,'encoder_checkpoint':'ccs_013_area_v4/full_wtw_seed20260922/checkpoint_best.pt'},args.output/'checkpoint_best.pt')
        else: stale+=1
        if epoch%10==0 or stale==0: print(json.dumps(record),flush=True)
        if stale>=args.patience: break
    (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n'); (args.output/'run_summary.json').write_text(json.dumps({'status':'complete','best_selection_score':best,'epochs_completed':len(history),'train_records':int(train.sum()),'validation_records':int(val.sum()),'test_accessed':False},indent=2)+'\n')
if __name__=='__main__': main()
