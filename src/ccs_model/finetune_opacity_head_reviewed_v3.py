#!/usr/bin/env python3
"""Low-LR reviewed-label fine-tuning with epoch-0 rollback protection."""

from __future__ import annotations
import argparse,json,math,random
from pathlib import Path
import torch
from torch.nn import functional as F
from sklearn.metrics import cohen_kappa_score
from ccs_model.metrics_v4 import multiclass_metrics
from ccs_model.train_document_v1 import OPACITY_NAMES
from ccs_model.train_opacity_features_v2 import Head

def load(feature_dir):
 chunks=[torch.load(p,map_location='cpu',weights_only=False) for p in sorted(feature_dir.glob('chunk_*.pt'))]; return torch.cat([c['features'] for c in chunks]),torch.cat([c['labels'] for c in chunks]),sum([c['splits'] for c in chunks],[])
def evaluate(model,x,y):
 model.eval()
 with torch.no_grad(): pred=model(x).argmax(1)
 m=multiclass_metrics(y.tolist(),pred.tolist(),OPACITY_NAMES); m['qwk']=float(cohen_kappa_score(y.tolist(),pred.tolist(),weights='quadratic')); return m
def main():
 p=argparse.ArgumentParser(); p.add_argument('--feature-dir',type=Path,required=True); p.add_argument('--base-checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--epochs',type=int,default=120); p.add_argument('--patience',type=int,default=20); p.add_argument('--seed',type=int,required=True); args=p.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed); x,y,splits=load(args.feature_dir); train=torch.tensor([s=='train' for s in splits]); val=torch.tensor([s=='val' for s in splits]); base=torch.load(args.base_checkpoint,map_location='cpu',weights_only=False); mean=base['feature_mean']; std=base['feature_std']; xt=(x[train]-mean)/std; xv=(x[val]-mean)/std; yt=y[train]; yv=y[val]
 model=Head(); model.load_state_dict(base['head_state_dict']); counts=torch.bincount(yt,minlength=4).float(); weights=counts.pow(-.35); weights/=weights.mean(); optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.10); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6); generator=torch.Generator().manual_seed(args.seed); args.output.mkdir(parents=True,exist_ok=False); initial=evaluate(model,xv,yv); best=.7*initial['macro_f1']+.3*initial['balanced_accuracy']; history=[{'epoch':0,'selection_score':best,'metrics':initial}]; torch.save({'architecture':'ccs.opacity-reviewed-finetune.v3','head_state_dict':model.state_dict(),'feature_mean':mean,'feature_std':std,'epoch':0,'metrics':initial},args.output/'checkpoint_best.pt'); stale=0
 for epoch in range(1,args.epochs+1):
  model.train(); order=torch.randperm(len(xt),generator=generator); losses=[]
  for start in range(0,len(order),32):
   idx=order[start:start+32]; feat=xt[idx]+.015*torch.randn(xt[idx].shape,generator=generator); target=yt[idx]; optimizer.zero_grad(set_to_none=True)
   if random.random()<.35 and len(idx)>1:
    perm=torch.randperm(len(idx),generator=generator); lam=random.uniform(.70,.90); logits=model(lam*feat+(1-lam)*feat[perm]); one=F.one_hot(target,4).float(); soft=lam*one+(1-lam)*one[perm]; sample_weight=(soft*weights).sum(1); loss=(-(soft*F.log_softmax(logits,1)).sum(1)*sample_weight).mean()
   else:
    logits=model(feat); ce=F.cross_entropy(logits,target,weight=weights,label_smoothing=.06); levels=torch.arange(4,dtype=logits.dtype); expected=(logits.softmax(1)*levels).sum(1); loss=ce+.15*F.smooth_l1_loss(expected,target.float(),beta=.75)
   loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); losses.append(float(loss.detach()))
  scheduler.step(); m=evaluate(model,xv,yv); score=.7*m['macro_f1']+.3*m['balanced_accuracy']; record={'epoch':epoch,'train_loss':sum(losses)/len(losses),'selection_score':score,'metrics':m}; history.append(record)
  if score>best:
   best=score; stale=0; torch.save({'architecture':'ccs.opacity-reviewed-finetune.v3','head_state_dict':model.state_dict(),'feature_mean':mean,'feature_std':std,'epoch':epoch,'metrics':m},args.output/'checkpoint_best.pt')
  else: stale+=1
  if epoch%10==0 or stale==0: print(json.dumps(record),flush=True)
  if stale>=args.patience: break
 (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n'); (args.output/'run_summary.json').write_text(json.dumps({'status':'complete','best_selection_score':best,'epochs_completed':len(history)-1,'test_accessed':False},indent=2)+'\n')
if __name__=='__main__':main()
