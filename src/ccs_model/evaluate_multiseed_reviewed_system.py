#!/usr/bin/env python3
"""Evaluate five matched opacity/area seeds and aggregate mean/std metrics."""

from __future__ import annotations
import argparse,json,re,statistics
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from ccs_model.metrics_v4 import area_metrics,multiclass_metrics
from ccs_model.train_area_features_v1 import AreaHead,load_data
from ccs_model.train_document_v1 import OPACITY_NAMES,CONTRACTION_NAMES
from ccs_model.train_opacity_features_v2 import Head

def contraction(a):return 0 if a>20 else 1 if a>16 else 2
def final_ccs(o,a):return 0 if o!=3 or a>20 else 1 if a>16 else 2
def seed_from(path):
 m=re.search(r'(20\d+)$',path.parent.name); return int(m.group(1)) if m else None
def aggregate(values):return {'mean':statistics.mean(values),'std':statistics.stdev(values) if len(values)>1 else 0.0,'values':values}
def main():
 p=argparse.ArgumentParser();p.add_argument('--feature-dir',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--opacity-root',type=Path,required=True);p.add_argument('--area-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 chunks=[torch.load(x,map_location='cpu',weights_only=False) for x in sorted(args.feature_dir.glob('chunk_*.pt'))];all_features=torch.cat([c['features'] for c in chunks]);labels=torch.cat([c['labels'] for c in chunks]);splits=sum([c['splits'] for c in chunks],[]);val=torch.tensor([s=='val' for s in splits]);features=all_features[val];opacity_ref=labels[val]
 x,area,wtw,train_mask,val_mask=load_data(args.feature_dir,args.manifest);xv=x[val_mask];area_ref=area[val_mask];wv=wtw[val_mask];con_ref=[contraction(v) for v in area_ref.tolist()];final_ref_base=opacity_ref.tolist()
 opacity_paths={seed_from(path):path for path in args.opacity_root.glob('feature_seed*/checkpoint_best.pt')};area_paths={seed_from(path):path for path in args.area_root.glob('area_seed*/checkpoint_best.pt')};seeds=sorted(set(opacity_paths)&set(area_paths));
 if len(seeds)!=5:raise ValueError(f'expected 5 matched seeds, found {seeds}')
 runs=[]
 for seed in seeds:
  oc=torch.load(opacity_paths[seed],map_location='cpu',weights_only=False);om=Head();om.load_state_dict(oc['head_state_dict']);om.eval();
  with torch.no_grad():op=om((features-oc['feature_mean'])/oc['feature_std']).softmax(1)
  opacity_pred=op.argmax(1).tolist();ac=torch.load(area_paths[seed],map_location='cpu',weights_only=False);am=AreaHead();am.load_state_dict(ac['head_state_dict']);am.eval();
  with torch.no_grad():area_pred=am(xv,wv)
  area_values=area_pred.tolist();con_pred=[contraction(v) for v in area_values];final_ref=[final_ccs(o,a) for o,a in zip(final_ref_base,area_ref.tolist(),strict=True)];final_pred=[final_ccs(o,a) for o,a in zip(opacity_pred,area_values,strict=True)];one=np.eye(4)[opacity_ref.numpy()]
  opacity_metrics=multiclass_metrics(opacity_ref.tolist(),opacity_pred,OPACITY_NAMES);con_metrics=multiclass_metrics(con_ref,con_pred,CONTRACTION_NAMES);final_metrics=multiclass_metrics(final_ref,final_pred,('negative','mild_positive','severe_positive'))
  runs.append({'seed':seed,'opacity':{'accuracy':opacity_metrics['exact_accuracy'],'balanced_accuracy':opacity_metrics['balanced_accuracy'],'macro_f1':opacity_metrics['macro_f1'],'macro_roc_auc_ovr':float(roc_auc_score(one,op.numpy(),average='macro',multi_class='ovr')),'grade3_sensitivity':opacity_metrics['per_class']['opacity_3']['recall_sensitivity'],'grade3_specificity':opacity_metrics['per_class']['opacity_3']['specificity']},'opening_area':area_metrics(area_ref.tolist(),area_values),'contraction':{'accuracy':con_metrics['exact_accuracy'],'balanced_accuracy':con_metrics['balanced_accuracy'],'macro_f1':con_metrics['macro_f1'],'grade23_sensitivity':con_metrics['per_class']['contraction_2_or_3']['recall_sensitivity'],'grade23_specificity':con_metrics['per_class']['contraction_2_or_3']['specificity']},'final_ccs':{'accuracy':final_metrics['exact_accuracy'],'balanced_accuracy':final_metrics['balanced_accuracy'],'macro_f1':final_metrics['macro_f1'],'severe_sensitivity':final_metrics['per_class']['severe_positive']['recall_sensitivity'],'severe_specificity':final_metrics['per_class']['severe_positive']['specificity']}})
 paths={'opacity':['accuracy','balanced_accuracy','macro_f1','macro_roc_auc_ovr','grade3_sensitivity','grade3_specificity'],'opening_area':['mae_mm2','rmse_mm2','signed_bias_mm2','within_1_mm2','within_2_mm2'],'contraction':['accuracy','balanced_accuracy','macro_f1','grade23_sensitivity','grade23_specificity'],'final_ccs':['accuracy','balanced_accuracy','macro_f1','severe_sensitivity','severe_specificity']};summary={section:{metric:aggregate([run[section][metric] for run in runs]) for metric in metrics} for section,metrics in paths.items()};report={'schema_version':'ccs.reviewed-multiseed.v1','seeds':seeds,'records':len(opacity_ref),'runs':runs,'summary':summary,'wtw_source_rule':'647 individual_or_embedded; 307 protocol_default_11.3_mm','test_accessed':False};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
