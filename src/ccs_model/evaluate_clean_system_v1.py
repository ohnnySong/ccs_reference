#!/usr/bin/env python3
"""Evaluate cleaned opacity ensemble plus WTW area ensemble and CCS rules."""

from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from ccs_model.metrics_v4 import area_metrics,multiclass_metrics
from ccs_model.train_area_features_v1 import AreaHead,load_data
from ccs_model.train_document_v1 import OPACITY_NAMES,CONTRACTION_NAMES
from ccs_model.train_opacity_features_v2 import Head

def contraction(area): return 0 if area>20 else 1 if area>16 else 2
def final_ccs(opacity,area): return 0 if opacity!=3 or area>20 else 1 if area>16 else 2

def main():
 p=argparse.ArgumentParser(); p.add_argument('--feature-dir',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--opacity-root',type=Path,required=True); p.add_argument('--area-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
 chunks=[torch.load(path,map_location='cpu',weights_only=False) for path in sorted(args.feature_dir.glob('chunk_*.pt'))]; all_features=torch.cat([c['features'] for c in chunks]); labels=torch.cat([c['labels'] for c in chunks]); splits=sum([c['splits'] for c in chunks],[]); val=torch.tensor([s=='val' for s in splits]); opacity_ref=labels[val]; feature=all_features[val]
 opacity_probs=[]; opacity_members=[]
 for path in sorted(args.opacity_root.glob('feature_seed*/checkpoint_best.pt')):
  c=torch.load(path,map_location='cpu',weights_only=False); m=Head(); m.load_state_dict(c['head_state_dict']); m.eval(); normalized=(feature-c['feature_mean'])/c['feature_std'];
  with torch.no_grad(): opacity_probs.append(m(normalized).softmax(1))
  opacity_members.append({'path':str(path),'epoch':c['epoch'],'macro_f1':c['metrics']['macro_f1']})
 opacity_probability=torch.stack(opacity_probs).mean(0); opacity_pred=opacity_probability.argmax(1)
 x,area,wtw,train_mask,val_mask=load_data(args.feature_dir,args.manifest); xv=x[val_mask]; area_ref=area[val_mask]; wv=wtw[val_mask]; area_predictions=[]; area_members=[]
 for path in sorted(args.area_root.glob('area_seed*/checkpoint_best.pt')):
  c=torch.load(path,map_location='cpu',weights_only=False); m=AreaHead(); m.load_state_dict(c['head_state_dict']); m.eval();
  with torch.no_grad(): area_predictions.append(m(xv,wv))
  area_members.append({'path':str(path),'epoch':c['epoch'],'mae_mm2':c['metrics']['mae_mm2']})
 area_pred=torch.stack(area_predictions).mean(0); contraction_ref=[contraction(x) for x in area_ref.tolist()]; contraction_pred=[contraction(x) for x in area_pred.tolist()]; final_ref=[final_ccs(o,a) for o,a in zip(opacity_ref.tolist(),area_ref.tolist(),strict=True)]; final_pred=[final_ccs(o,a) for o,a in zip(opacity_pred.tolist(),area_pred.tolist(),strict=True)]; one=np.eye(4)[opacity_ref.numpy()]
 report={'schema_version':'ccs.cleaned-system.v1','records':len(area_ref),'opacity_members':opacity_members,'area_members':area_members,'opacity':{'metrics':multiclass_metrics(opacity_ref.tolist(),opacity_pred.tolist(),OPACITY_NAMES),'macro_roc_auc_ovr':float(roc_auc_score(one,opacity_probability.numpy(),average='macro',multi_class='ovr'))},'opening_area':area_metrics(area_ref.tolist(),area_pred.tolist()),'contraction_0_1_23':multiclass_metrics(contraction_ref,contraction_pred,CONTRACTION_NAMES),'final_ccs':multiclass_metrics(final_ref,final_pred,('negative','mild_positive','severe_positive')),'wtw_source_rule':'647 individual_or_embedded; 307 protocol_default_11.3_mm','test_accessed':False}
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
