#!/usr/bin/env python3
"""Extract frozen CCS encoder features in short, crash-resilient chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ccs_model.opacity_regularized_v2 import OpacityRegularizedV2
from ccs_model.train_opacity_v2 import SafeOpacityDataset


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--encoder-checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--start',type=int,required=True); p.add_argument('--end',type=int,required=True); p.add_argument('--device',choices=('cpu','cuda'),default='cuda'); args=p.parse_args()
    rows=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    selected=rows[args.start:args.end]
    dataset=SafeOpacityDataset(selected,args.data_root,size=320,train=False)
    device=torch.device(args.device)
    if device.type=='cpu': torch.set_num_threads(4)
    model=OpacityRegularizedV2().to(device).eval()
    source=torch.load(args.encoder_checkpoint,map_location='cpu',weights_only=False)
    state={k.removeprefix('encoder.'):v for k,v in source['state_dict'].items() if k.startswith('encoder.')}
    model.encoder.load_state_dict(state,strict=True)
    features=[]; labels=[]; splits=[]
    with torch.no_grad():
        for index in range(len(dataset)):
            item=dataset[index]
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                feature=model.extract_features(item['image'].unsqueeze(0).to(device))
            features.append(feature.float().cpu())
            labels.append(int(item['opacity']))
            splits.append(str(selected[index]['split']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'start':args.start,'end':args.end,'features':torch.cat(features),'labels':torch.tensor(labels),'splits':splits},args.output)
    print(args.output,len(selected),flush=True)
if __name__=='__main__': main()
