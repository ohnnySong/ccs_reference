#!/usr/bin/env python3
"""Evaluate a probability ensemble of regularized opacity feature heads."""

from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from ccs_model.metrics_v4 import multiclass_metrics
from ccs_model.train_document_v1 import OPACITY_NAMES
from ccs_model.train_opacity_features_v2 import Head

def main():
 p=argparse.ArgumentParser(); p.add_argument('--feature-dir',type=Path,required=True); p.add_argument('--runs-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
 chunks=[torch.load(x,map_location='cpu',weights_only=False) for x in sorted(args.feature_dir.glob('chunk_*.pt'))]; x=torch.cat([c['features'] for c in chunks]); y=torch.cat([c['labels'] for c in chunks]); splits=sum([c['splits'] for c in chunks],[]); val=torch.tensor([s=='val' for s in splits]); x=x[val]; y=y[val]
 probs=[]; members=[]
 for path in sorted(args.runs_root.glob('feature_seed*/checkpoint_best.pt')):
  c=torch.load(path,map_location='cpu',weights_only=False); model=Head(); model.load_state_dict(c['head_state_dict']); model.eval(); normalized=(x-c['feature_mean'])/c['feature_std']
  with torch.no_grad(): probs.append(model(normalized).softmax(1))
  members.append({'path':str(path),'epoch':c['epoch'],'macro_f1':c['metrics']['macro_f1']})
 mean=torch.stack(probs).mean(0); pred=mean.argmax(1); metrics=multiclass_metrics(y.tolist(),pred.tolist(),OPACITY_NAMES); one=np.eye(4)[y.numpy()]
 report={'schema_version':'ccs.opacity-ensemble.v2','members':members,'validation':metrics,'macro_roc_auc_ovr':float(roc_auc_score(one,mean.numpy(),average='macro',multi_class='ovr')),'validation_records':len(y),'test_accessed':False}
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
